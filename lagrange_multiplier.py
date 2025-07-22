# Solve Poisson's equation subject to the constrain that the
# solution takes a given value on closed surfaces embedded in
# the domain. Also see
# https://www.dealii.org/current/doxygen/deal.II/step_60.html
# NOTE: the Schur complement behaves like a Neumann-to-Dirichlet
# map, which is important for designing a good preconditioner

import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, div
from mpi4py import MPI
from petsc4py import PETSc
from utils import norm_L2, one_sided_int_entities
from dolfinx.fem.petsc import LinearProblem
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

# Create sub-mesh for Lagrange multiplier. We locate the facets on the
# interface (gamma_1) pass them to create_submesh
tdim = msh.topology.dim
fdim = tdim - 1
gamma_i_facets = ft.find(bound_ids["gamma_i"])
submesh, submesh_emap = mesh.create_submesh(msh, fdim, gamma_i_facets)[0:2]

# Create functions spaces
V = fem.functionspace(msh, ("Lagrange", k))
W = fem.functionspace(submesh, ("Lagrange", k))
X = ufl.MixedFunctionSpace(V, W)

# Trial and test functions
u, lmbda = ufl.TrialFunctions(X)
v, eta = ufl.TestFunctions(X)

# Create Dirichlet boundary condition
msh.topology.create_entities(fdim)
dirichlet_facets = ft.find(bound_ids["gamma"])
dirichlet_dofs = fem.locate_dofs_topological(V, fdim, dirichlet_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dirichlet_dofs, V)

# We take `msh`` to be the integration domain mesh (we will pass this mesh the
# domain when creating measures). We must provide entity maps relating this
# mesh to the other meshes in the form (here just `submesh`)
entity_maps = [submesh_emap]

# Create integration measure for the interface terms. We specify the facets
# on gamma_i, which are identified as (cell, local facet index) pairs
facet_integration_entities = one_sided_int_entities(msh, gamma_i_facets)
ds = ufl.Measure(
    "ds",
    subdomain_data=[(bound_ids["gamma_i"], facet_integration_entities)],
    domain=msh,
)

a = (
    inner(grad(u), grad(v)) * ufl.dx
    + inner(lmbda, v) * ds(bound_ids["gamma_i"])
    + inner(u, eta) * ds(bound_ids["gamma_i"])
)

x_msh = ufl.SpatialCoordinate(msh)
x_sm = ufl.SpatialCoordinate(submesh)
f = -div(grad(u_e(x_msh)))

L = inner(f, v) * ufl.dx + inner(u_e(x_sm), eta) * ufl.dx

u, lmbda = fem.Function(V), fem.Function(W)
petsc_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "superlu_dist"}
problem = LinearProblem(
    ufl.extract_blocks(a),
    ufl.extract_blocks(L),
    u=[u, lmbda],
    bcs=[bc],
    kind="mpi",
    petsc_options_prefix="lagrange_multiplier_",
    petsc_options=petsc_opts,
    entity_maps=entity_maps,
)
problem.solve()

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
