import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, div
from mpi4py import MPI
from petsc4py import PETSc
from utils import reorder_mesh


def norm_L2(comm, v):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(inner(v, v) * ufl.dx)), op=MPI.SUM))


n = 8
assert n % 2 == 0  # NOTE n must be even
k = 1
msh = mesh.create_unit_square(
    MPI.COMM_WORLD, n, n, ghost_mode=mesh.GhostMode.none)
# msh = mesh.create_unit_cube(
#     MPI.COMM_WORLD, n, n, n, ghost_mode=mesh.GhostMode.none)

# Currently, permutations are not working in parallel, so reorder the mesh
reorder_mesh(msh)

V = fem.FunctionSpace(msh, ("Lagrange", k))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Create Dirichlet boundary condition
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_entities(fdim)
dirichlet_facets = mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)))
dirichlet_dofs = fem.locate_dofs_topological(V, fdim, dirichlet_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dirichlet_dofs, V)

# Create submesh of centre facets
centre_facets = mesh.locate_entities(
    msh, fdim, lambda x: np.isclose(x[0], 0.5))
submesh, entity_map = mesh.create_submesh(msh, fdim, centre_facets)[0:2]

# Create function space for the Lagrange multiplier
W = fem.FunctionSpace(submesh, ("Lagrange", k))
lmbda = ufl.TrialFunction(W)
eta = ufl.TestFunction(W)

facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
entity_maps = {submesh: [entity_map.index(entity)
                         if entity in entity_map else -1
                         for entity in range(num_facets)]}

# Create measure for integration
facet_integration_entities = {1: []}
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
for facet in centre_facets:
    # Check if this facet is owned
    if facet < facet_imap.size_local:
        # Get a cell connected to the facet
        cell = f_to_c.links(facet)[0]
        local_facet = c_to_f.links(cell).tolist().index(facet)
        facet_integration_entities[1].extend([cell, local_facet])
ds = ufl.Measure("ds", subdomain_data=facet_integration_entities, domain=msh)


def u_e(x):
    return x[0] * (1 - x[0])


# Define forms
a_00 = fem.form(inner(grad(u), grad(v)) * ufl.dx)
a_01 = fem.form(inner(lmbda, v) * ds(1), entity_maps=entity_maps)
a_10 = fem.form(inner(u, eta) * ds(1), entity_maps=entity_maps)

# f = fem.Constant(msh, PETSc.ScalarType(2.0))
x_msh = ufl.SpatialCoordinate(msh)
f = - div(grad(u_e(x_msh)))
L_0 = fem.form(inner(f, v) * ufl.dx)

# c = fem.Constant(submesh, PETSc.ScalarType(0.25))
x_sm = ufl.SpatialCoordinate(submesh)
L_1 = fem.form(inner(u_e(x_sm), eta) * ufl.dx)

a = [[a_00, a_01],
     [a_10, None]]
L = [L_0, L_1]

# Use block assembly
A = fem.petsc.assemble_matrix_block(a, bcs=[bc])
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=[bc])

# Configure solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecLeft()
ksp.solve(b, x)

# Recover solution
u, lmbda = fem.Function(V), fem.Function(W)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
u.x.scatter_forward()
lmbda.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
lmbda.x.scatter_forward()

# Write to file
with io.VTXWriter(msh.comm, "u.bp", u) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "lmbda.bp", lmbda) as f:
    f.write(0.0)

# Compute L^2-norm of error
e_L2 = norm_L2(msh.comm, u - u_e(x_msh))
rank = msh.comm.Get_rank()
if rank == 0:
    print(f"e_L2 = {e_L2}")
