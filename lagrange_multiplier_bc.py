# Solves u - div(grad(u)) = f, where the Dirichlet boundary condition is
# enforced via a Lagrange multiplier. See "The finite element method with
# Lagrangian multipliers" by Babuška (1973)

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, div
from mpi4py import MPI
from petsc4py import PETSc
from utils import norm_L2


# Marker for the domain boundary
def boundary_marker(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], l_x)),
                         np.logical_or(np.isclose(x[1], 0.0),
                                       np.isclose(x[1], l_y)))


# Create mesh
l_x = 2.0
l_y = 1.0
n_x = 16
n_y = 8
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, points=((0.0, 0.0), (l_x, l_y)), n=(n_x, n_y))

# Create sub-mesh of the boundary to define function space for the Lagrange
# multipiler
tdim = msh.topology.dim
fdim = tdim - 1
num_facets = msh.topology.create_entities(fdim)
boundary_facets = mesh.locate_entities_boundary(
    msh, fdim, boundary_marker)
submesh, submesh_to_mesh = mesh.create_submesh(msh, fdim, boundary_facets)[0:2]

# Create function spaces on the mesh and sub-mesh
k = 3  # Polynomial degree
V = fem.FunctionSpace(msh, ("Lagrange", k))
W = fem.FunctionSpace(submesh, ("Lagrange", k))

# Trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
lmbda = ufl.TrialFunction(W)
mu = ufl.TestFunction(W)

# Create manufactured solution
x = ufl.SpatialCoordinate(msh)
u_e = 1
for i in range(tdim):
    u_e *= ufl.cos(ufl.pi * x[i])
f = u_e - div(grad(u_e))

# Dirichlet boundary condition (enforced through Lagrange multiplier)
u_d = u_e

# Create measure for integral over boundary. We take the integration domain
# mesh to be msh
ds = ufl.Measure("ds", domain=msh)

# Since the integration domain is msh, we must provide a map from facets
# in msh to cells in submesh. This is simply the "inverse" of
# submesh_to_mesh and can be computed as follows:
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
mesh_to_submesh = np.full(num_facets, -1)
mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
entity_maps = {submesh: mesh_to_submesh}

# Define forms
a_00 = fem.form(inner(u, v) * ufl.dx + inner(grad(u), grad(v)) * ufl.dx)
a_01 = fem.form(- inner(lmbda, v) * ds, entity_maps=entity_maps)
a_10 = fem.form(- inner(u, mu) * ds, entity_maps=entity_maps)
a_11 = None
L_0 = fem.form(inner(f, v) * ufl.dx)
L_1 = fem.form(- inner(u_d, mu) * ds, entity_maps=entity_maps)

# Define block structure
a = [[a_00, a_01],
     [a_10, a_11]]
L = [L_0, L_1]

# Assemble matrices
A = fem.petsc.assemble_matrix_block(a)
A.assemble()
b = fem.petsc.assemble_vector_block(L, a)

# Solve
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
e_L2 = norm_L2(msh.comm, u - u_e)

if msh.comm.rank == 0:
    print(f"e_L2 = {e_L2}")
