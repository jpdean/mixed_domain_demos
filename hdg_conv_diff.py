# Solves the convection-diffusion equation using the HDG scheme from
# https://epubs.siam.org/doi/10.1137/090775464


from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from utils import norm_L2, compute_cell_boundary_int_entities
from dolfinx.fem.petsc import LinearProblem


def u_e(x):
    "Function to represent the exact solution"
    if isinstance(x, ufl.SpatialCoordinate):
        module = ufl
    else:
        module = np

    return module.sin(3.0 * module.pi * x[0]) * module.cos(2.0 * module.pi * x[1])


def boundary(x):
    "A function to mark the domain boundary"
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    )


# Create a mesh
comm = MPI.COMM_WORLD
n = 16
msh = mesh.create_unit_square(comm, n, n)

# Create a sub-mesh of all facets in the mesh to allow the facet function
# spaces to be created
tdim = msh.topology.dim
fdim = tdim - 1
num_cell_facets = cell_num_entities(msh.topology.cell_type, fdim)
msh.topology.create_entities(fdim)
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)
# NOTE Despite all facets being present in the submesh, the entity map isn't
# necessarily the identity in parallel
facet_mesh, facet_mesh_emap = mesh.create_submesh(msh, fdim, facets)[0:2]

# Create functions spaces
k = 3  # Polynomial degree
V = fem.functionspace(msh, ("Discontinuous Lagrange", k))
Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k))
W = ufl.MixedFunctionSpace(V, Vbar)

# Create trial and test functions
u, ubar = ufl.TrialFunctions(W)
v, vbar = ufl.TestFunctions(W)

# Create integration entities and define integration measures. We want
# to integrate around each element boundary, so we call the following
# convenience function:
cell_boundary_facets = compute_cell_boundary_int_entities(msh)
dx_c = ufl.Measure("dx", domain=msh)
cell_boundaries = 0  # Tag
ds_c = ufl.Measure("ds", subdomain_data=[(cell_boundaries, cell_boundary_facets)], domain=msh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

# Create entity maps. We take msh to be the integration domain, so we must
# provide a map relating entities in `msh` to entities in `facet_mesh`
entity_maps = [facet_mesh_emap]

# Define finite element forms
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
kappa = fem.Constant(msh, PETSc.ScalarType(1e-3))
gamma = 16.0 * k**2 / h

# Diffusive terms
a = (
    inner(kappa * grad(u), grad(v)) * dx_c
    - inner(kappa * (u - ubar), dot(grad(v), n)) * ds_c(cell_boundaries)
    - inner(dot(grad(u), n), kappa * (v - vbar)) * ds_c(cell_boundaries)
    + gamma * inner(kappa * (u - ubar), v - vbar) * ds_c(cell_boundaries)
)

# Advection terms
x = ufl.SpatialCoordinate(msh)
w = ufl.as_vector(
    (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
    )
)
lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), 0, 1)
a += -inner(w * u, grad(v)) * dx_c + inner(dot(w * (u - lmbda * (u - ubar)), n), v - vbar) * ds_c(
    cell_boundaries
)

# RHS
f = dot(w, grad(u_e(x))) - div(kappa * grad(u_e(x)))
L = inner(f, v) * dx_c + inner(fem.Constant(facet_mesh, 0.0), vbar) * dx_f

# Define the boundary condition. We begin by locating the facets on the
# domain boundary
msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
# Since Vbar is defined over facet_mesh, we must find the cells in
# facet_mesh corresponding to msh_boundary_facets
facet_mesh_boundary_facets = entity_maps[0].sub_topology_to_topology(
    msh_boundary_facets, inverse=True
)
# We can now use these facets to locate the desired DOFs
facet_mesh.topology.create_connectivity(fdim, fdim)
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
# Finally, we interpolate the boundary condition
u_bc = fem.Function(Vbar)
u_bc.interpolate(u_e)
bc = fem.dirichletbc(u_bc, dofs)
bcs = [bc]

u, ubar = fem.Function(V), fem.Function(Vbar)
petsc_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "superlu_dist"}
problem = LinearProblem(
    ufl.extract_blocks(a),
    ufl.extract_blocks(L),
    u=[u, ubar],
    bcs=bcs,
    kind="mpi",
    petsc_options_prefix="hdg_conv_diff_",
    petsc_options=petsc_opts,
    entity_maps=entity_maps,
)
problem.solve()

# Write solution to file
with io.VTXWriter(msh.comm, "u.bp", u) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "ubar.bp", ubar) as f:
    f.write(0.0)

# Compute the error
x = ufl.SpatialCoordinate(msh)
e_L2 = norm_L2(msh.comm, u - u_e(x))

if comm.rank == 0:
    print(f"e_L2 = {e_L2}")
