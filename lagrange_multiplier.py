# Solve Poisson's equation subject to the constrain that the
# solution takes a given value on closed surfaces embedded in
# the domain. Also see
# https://www.dealii.org/current/doxygen/deal.II/step_60.html
# NOTE: the Schur complement behaves like a Neumann-to-Dirichlet
# map, which is important for designing a good preconditioner

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, div
from mpi4py import MPI
from petsc4py import PETSc
from utils import norm_L2
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from meshing import create_fenics_logo_msh, create_box_with_sphere_msh


def u_e(x):
    "Function to represent the exact solution"
    u_e = 1
    for i in range(tdim):
        u_e *= ufl.sin(ufl.pi * x[i])
    return u_e


# Set some paramters
comm = MPI.COMM_WORLD
d = 2  # Geometric dimension
h = 0.05  # Max cell diameter
k = 3  # Polynomial degree

# Create trial and test functions for primary unknown
if d == 2:
    msh, ct, ft, vol_ids, bound_ids = create_fenics_logo_msh(comm, h)
else:
    assert d == 3
    msh, ct, ft, vol_ids, bound_ids = create_box_with_sphere_msh(comm, h)

V = fem.functionspace(msh, ("Lagrange", k))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Create Dirichlet boundary condition
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_entities(fdim)
dirichlet_facets = ft.find(bound_ids["gamma"])
dirichlet_dofs = fem.locate_dofs_topological(V, fdim, dirichlet_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dirichlet_dofs, V)

# Create sub-mesh for Lagrange multiplier. We locate the facets on the
# interface (gamma_1) pass them to create_submesh
gamma_i_facets = ft.find(bound_ids["gamma_i"])
submesh, submesh_to_mesh = mesh.create_submesh(msh, fdim, gamma_i_facets)[0:2]

# Create function space for the Lagrange multiplier
W = fem.functionspace(submesh, ("Lagrange", k))
lmbda, eta = ufl.TrialFunction(W), ufl.TestFunction(W)

# We take msh to be the integration domain mesh, so we must provide a map
# from facets in msh to cells in submesh. This is simply the "inverse" of
# submesh_to_mesh and can be computed as follows.
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
msh_to_submesh = np.full(num_facets, -1)
msh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
entity_maps = {submesh: msh_to_submesh}

# Create integration measure for the interface terms. We specify the facets
# on gamma_i, which are identified as (cell, local facet index) pairs
facet_integration_entities = []
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
# Loop through all interface facets
for facet in gamma_i_facets:
    # Check if this facet is owned
    if facet < facet_imap.size_local:
        # Get a cell connected to the facet
        cell = f_to_c.links(facet)[0]
        local_facet = np.where(c_to_f.links(cell) == facet)[0][0]
        facet_integration_entities.extend([cell, local_facet])
ds = ufl.Measure(
    "ds",
    subdomain_data=[(bound_ids["gamma_i"], facet_integration_entities)],
    domain=msh,
)

# Define forms
a_00 = fem.form(inner(grad(u), grad(v)) * ufl.dx)
a_01 = fem.form(inner(lmbda, v) * ds(bound_ids["gamma_i"]), entity_maps=entity_maps)
a_10 = fem.form(inner(u, eta) * ds(bound_ids["gamma_i"]), entity_maps=entity_maps)

x_msh = ufl.SpatialCoordinate(msh)
f = -div(grad(u_e(x_msh)))
# f = fem.Constant(msh, PETSc.ScalarType(2.0))
L_0 = fem.form(inner(f, v) * ufl.dx)

x_sm = ufl.SpatialCoordinate(submesh)
L_1 = fem.form(inner(u_e(x_sm), eta) * ufl.dx)
# L_1 = fem.form(inner(0.25 + 0.3 * ufl.sin(ufl.pi * x_sm[0]), eta) * ufl.dx)

# Define block structure
a = [[a_00, a_01], [a_10, None]]
L = [L_0, L_1]

# Assemble matrix
A = assemble_matrix_block(a, bcs=[bc])
A.assemble()

# Assemble vector
b = assemble_vector_block(L, a, bcs=[bc])

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
lmbda.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
lmbda.x.scatter_forward()

# Write to file
with io.VTXWriter(msh.comm, "u.bp", u, "BP4") as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "lmbda.bp", lmbda, "BP4") as f:
    f.write(0.0)

# Compute L^2-norm of error
e_L2 = norm_L2(msh.comm, u - u_e(x_msh))
rank = msh.comm.Get_rank()
if rank == 0:
    print(f"e_L2 = {e_L2}")
