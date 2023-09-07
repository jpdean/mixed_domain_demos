# Solves the convection-diffusion equation using the HDG scheme from
# https://epubs.siam.org/doi/10.1137/090775464


from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from utils import norm_L2


def u_e(x):
    "Function to represent the exact solution"
    if type(x) == ufl.SpatialCoordinate:
        module = ufl
    else:
        module = np

    return module.sin(3.0 * module.pi * x[0]) * \
        module.cos(2.0 * module.pi * x[1])


def boundary(x):
    "A function to mark the domain boundary"
    lr = np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    tb = np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    return lr | tb


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
facet_mesh, facet_mesh_to_msh = mesh.create_submesh(msh, fdim, facets)[0:2]

# Create functions spaces
k = 3  # Polynomial degree
V = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
Vbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

# Create trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
ubar, vbar = ufl.TrialFunction(Vbar), ufl.TestFunction(Vbar)

# Create integration entities and define integration measures. We want
# to integrate around each element boundary so we loop over cells and
# add each facet of the cells as (cell, local facet) pairs
# TODO Create without Python loop using numpy to improve performance
all_facets = 0  # Tag
facet_integration_entities = []
for cell in range(msh.topology.index_map(tdim).size_local):
    for local_facet in range(num_cell_facets):
        facet_integration_entities.extend([cell, local_facet])
dx_c = ufl.Measure("dx", domain=msh)
ds_c = ufl.Measure("ds",
                   subdomain_data=[(all_facets, facet_integration_entities)],
                   domain=msh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

# Create entity maps. We take msh to be the integration domain, so the
# entity maps must map from facets in msh to cells in facet_mesh. This
# is the "inverse" of facet_mesh_to_msh.
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
msh_to_facet_mesh = np.full(num_facets, -1)
msh_to_facet_mesh[facet_mesh_to_msh] = np.arange(len(facet_mesh_to_msh))
entity_maps = {facet_mesh: msh_to_facet_mesh}

# Define finite element forms
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
kappa = fem.Constant(msh, PETSc.ScalarType(1e-3))
gamma = 16.0 * k**2 / h

# Diffusive terms
a_00 = inner(kappa * grad(u), grad(v)) * dx_c \
    - inner(kappa * dot(grad(u), n), v) * ds_c(all_facets) \
    - inner(kappa * u, dot(grad(v), n)) * ds_c(all_facets) \
    + gamma * inner(kappa * u, v) * ds_c(all_facets)
a_01 = inner(kappa * ubar, dot(grad(v), n)) * ds_c(all_facets) \
    - gamma * inner(kappa * ubar, v) * ds_c(all_facets)
a_10 = inner(kappa * dot(grad(u), n), vbar) * ds_c(all_facets) \
    - gamma * inner(kappa * u, vbar) * ds_c(all_facets)
a_11 = gamma * inner(kappa * ubar, vbar) * ds_c(all_facets)

# Advection terms
x = ufl.SpatialCoordinate(msh)
w = ufl.as_vector(
    (ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
     ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])))
lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), 0, 1)
a_00 += - inner(w * u, grad(v)) * dx_c \
    + inner(dot(w * u, n), v) * ds_c(all_facets) \
    - inner(lmbda * dot(w * u, n), v) * ds_c(all_facets)
a_01 += inner(lmbda * dot(w * ubar, n), v) * ds_c(all_facets)
a_10 += - inner(dot(w * u, n), vbar) * ds_c(all_facets) \
    + inner(lmbda * dot(w * u, n), vbar) * ds_c(all_facets)
a_11 += - inner(lmbda * dot(w * ubar, n), vbar) * ds_c(all_facets)

# Compile forms
a_00 = fem.form(a_00)
a_01 = fem.form(a_01, entity_maps=entity_maps)
a_10 = fem.form(a_10, entity_maps=entity_maps)
a_11 = fem.form(a_11, entity_maps=entity_maps)

f = dot(w, grad(u_e(x))) - div(kappa * grad(u_e(x)))

L_0 = fem.form(inner(f, v) * dx_c)
L_1 = fem.form(inner(fem.Constant(facet_mesh, 0.0), vbar) * dx_f)

# Define block structure
a = [[a_00, a_01],
     [a_10, a_11]]
L = [L_0, L_1]

# Define the boundary condition. We begin by locating the facets on the
# domain boundary
msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
# Since Vbar is defined over facet_mesh, we must find the cells in
# facet_mesh corresponding to msh_boundary_facets
facet_mesh_boundary_facets = msh_to_facet_mesh[msh_boundary_facets]
# We can now use these facets to locate the desired DOFs
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
# Finally, we interpolate the boundary condition
u_bc = fem.Function(Vbar)
u_bc.interpolate(u_e)
bc = fem.dirichletbc(u_bc, dofs)

# Assemble system of equations
A = fem.petsc.assemble_matrix_block(a, bcs=[bc])
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=[bc])

# Setup solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

# Recover the solution
u = fem.Function(V)
ubar = fem.Function(Vbar)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
ubar.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
u.x.scatter_forward()
ubar.x.scatter_forward()

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
