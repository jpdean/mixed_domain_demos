# This demo shows how Neumann boundary conditions can be interpolated
# into function spaces defined only over the Neumann boundary. This
# provides a more natural representation of boundary conditions and is
# more computationally efficient.


import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, div, dot
from mpi4py import MPI
from petsc4py import PETSc
from utils import norm_L2
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


def boundary_marker(x):
    "A function to mark the domain boundary"
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)),
                         np.logical_or(np.isclose(x[1], 0.0),
                                       np.isclose(x[1], 1.0)))


# Create a mesh and a sub-mesh of the boundary
n = 8
msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
tdim = msh.topology.dim
fdim = tdim - 1
num_facets = msh.topology.create_entities(fdim)
boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary_marker)
submesh, submesh_to_mesh = mesh.create_submesh(msh, fdim, boundary_facets)[0:2]

# Create function spaces
k = 3  # Polynomial degree
V = fem.functionspace(msh, ("Lagrange", k))
W = fem.functionspace(submesh, ("Lagrange", k))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Create integration measure and entity maps
ds = ufl.Measure("ds", domain=msh)
# We take msh to be the integration domain, so we must provide a map from
# facets in msh to cells in submesh. This is simply the "inverse" of
# submesh_to_mesh
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
msh_to_submesh = np.full(num_facets, -1)
msh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
entity_maps = {submesh: msh_to_submesh}

# Create manufactured solution
x = ufl.SpatialCoordinate(msh)
u_e = 1
for i in range(tdim):
    u_e *= ufl.sin(ufl.pi * x[i])
f = u_e - div(grad(u_e))
n = ufl.FacetNormal(msh)
g = dot(grad(u_e), n)

# Define forms
a = fem.form(inner(u, v) * ufl.dx + inner(grad(u), grad(v)) * ufl.dx)
L = fem.form(inner(f, v) * ufl.dx + inner(g, v) * ds, entity_maps=entity_maps)

# Assemble matrix and vector
A = assemble_matrix(a)
A.assemble()
b = assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Create solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Solve
u = fem.Function(V)
ksp.solve(b, u.vector)
u.x.scatter_forward()

# Write to file
with io.VTXWriter(msh.comm, "u.bp", u, "BP4") as f:
    f.write(0.0)

# Compute the error
e_L2 = norm_L2(msh.comm, u - u_e)

if msh.comm.rank == 0:
    print(f"e_L2 = {e_L2}")
