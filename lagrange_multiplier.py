# TODO This probably needs dof without cell fix

from threading import local
import numpy as np
import ufl
from dolfinx import fem, io, mesh, graph
from ufl import grad, inner
from mpi4py import MPI
from petsc4py import PETSc
from utils import reorder_mesh


def norm_L2(comm, v):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(inner(v, v) * ufl.dx)), op=MPI.SUM))


n = 4
assert n % 2 == 0  # NOTE n must be even
k = 1
msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=mesh.GhostMode.none)
# msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=mesh.GhostMode.none)

reorder_mesh(msh)

V = fem.FunctionSpace(msh, ("Lagrange", k))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Create Dirichlet boundary condition
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_entities(fdim)
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
dirichlet_facets = mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)))
dirichlet_dofs = fem.locate_dofs_topological(V, fdim, dirichlet_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dirichlet_dofs, V)

# # FIXME Need to use this clumsy method until we have better support for one
# # sided integrals
# # Create a submesh of the cells on the left side of the mesh
# left_cells = mesh.locate_entities(
#     msh, tdim, lambda x: x[0] <= 0.5)
# submesh_left_cells, entity_map_left_cells = mesh.create_submesh(
#     msh, tdim, left_cells)[0:2]
# with io.XDMFFile(submesh_left_cells.comm, "submesh_left_cells.xdmf", "w") as file:
#     file.write_mesh(submesh_left_cells)

# # Tag the facets on the right boundary of the submesh. These correspond to
# # the centre facets of the original mesh
# # NOTE Numbered with respect to submesh_left_cells
# centre_facets = mesh.locate_entities_boundary(
#     submesh_left_cells, fdim, lambda x: np.isclose(x[0], 0.5))
# mt = mesh.meshtags(
#     submesh_left_cells, fdim, centre_facets, 1)

# # Create an exterior facet measure on the submesh using the meshtags. Forms
# # with ds(1) therefore integrates over the centre facets of the orignial mesh
# ds = ufl.Measure("ds", domain=submesh_left_cells, subdomain_data=mt)

# # Create a submesh of the centre facets of the mesh to define the function space
# # for the Lagrange multiplier
# submesh_centre_facets, entity_map_centre_facets = mesh.create_submesh(
#     submesh_left_cells, fdim, centre_facets)[0:2]
# with io.XDMFFile(submesh_centre_facets.comm,
#                  "submesh_centre_facets.xdmf", "w") as file:
#     file.write_mesh(submesh_centre_facets)

# # We need to provide entitiy maps for both the original mesh and the submesh
# # of the centre facets
# facet_imap = submesh_left_cells.topology.index_map(fdim)
# sm_lc_num_facets = facet_imap.size_local + facet_imap.num_ghosts
# entity_maps = {msh: entity_map_left_cells,
#                submesh_centre_facets: [entity_map_centre_facets.index(entity)
#                                        if entity in entity_map_centre_facets else -1
#                                        for entity in range(sm_lc_num_facets)]}
# # END OF CLUMSY METHOD

centre_facets = mesh.locate_entities(
    msh, fdim, lambda x: np.isclose(x[0], 0.5))
submesh, entity_map = mesh.create_submesh(msh, fdim, centre_facets)[0:2]

# with io.XDMFFile(msh.comm, "msh.xdmf", "w") as file:
#     file.write_mesh(msh)
# with io.XDMFFile(msh.comm, "submesh.xdmf", "w") as file:
#     file.write_mesh(submesh)

# Create function space for the Lagrange multiplier
W = fem.FunctionSpace(submesh, ("Lagrange", k))
lmbda = ufl.TrialFunction(W)
eta = ufl.TestFunction(W)

entity_maps = {submesh: [entity_map.index(entity)
                         if entity in entity_map else -1
                         for entity in range(num_facets)]}

facet_integration_entities = {1: []}
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
for facet in centre_facets:
    # Check if this facet is owned
    if facet < facet_imap.size_local:
        # Get a cell
        cell = f_to_c.links(facet)[0]
        local_facet = c_to_f.links(cell).tolist().index(facet)
        facet_integration_entities[1].extend([cell, local_facet])

ds = ufl.Measure("ds", subdomain_data=facet_integration_entities, domain=msh)

# Define forms
a_00 = fem.form(inner(grad(u), grad(v)) * ufl.dx)
a_01 = fem.form(inner(lmbda, v) * ds(1), entity_maps=entity_maps)
a_10 = fem.form(inner(u, eta) * ds(1), entity_maps=entity_maps)
f = fem.Constant(msh, PETSc.ScalarType(2.0))
L_0 = fem.form(inner(f, v) * ufl.dx)
c = fem.Constant(submesh, PETSc.ScalarType(0.25))
L_1 = fem.form(inner(c, eta) * ufl.dx)
# x = ufl.SpatialCoordinate(submesh)
# L_1 = fem.form(inner(- 0.1 * ufl.sin(ufl.pi * x[1]), eta) * ufl.dx)

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
with io.VTXWriter(msh.comm, "poisson_lm_u.bp", u) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "poisson_lm_lmbda.bp", lmbda) as f:
    f.write(0.0)

# Compute L^2-norm of error
x = ufl.SpatialCoordinate(msh)
u_e = x[0] * (1 - x[0])
e_L2 = norm_L2(msh.comm, u - u_e)
rank = msh.comm.Get_rank()
if rank == 0:
    print(f"e_L2 = {e_L2}")
