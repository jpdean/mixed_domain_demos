# Scheme based on "A finite element method for domain decomposition
# with non-matching grids" by Becker et al. but with a DG scheme to
# solve the advection diffusion equation on half of the domain, and
# a standard CG Poisson solver on the other half.

# Consider a square domain on which we wish to solve the
# advection diffusion equations. The velocity field is given by
# (0.5 - x_1, 0.0) in the bottom half of the domain, and (0.0, 0.0)
# in the top half. We solve the bottom half of the domain with a
# DG advection-diffusion solver, and the top half with a standard
# CG solver. We enforce the Dirichlet boundary condition weakly
# for the DG scheme and strongly for the CG scheme. The assumed
# solution is u = sin(\pi * x_0) * sin(\pi * x_1). In this problem,
# the bottom half can be thought of as a fluid and the top half
# a solid, and the unknown u is the temperature field.

# NOTE: Since the velocity goes to zero at the interface x[1] = 0.5,
# the coupling is due only to the diffusion. No advective interface
# terms have been added

from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, avg, div, jump
import numpy as np
from petsc4py import PETSc
from utils import (
    norm_L2,
    convert_facet_tags,
    interface_int_entities,
    interior_facet_int_entities,
)
from dolfinx.fem.petsc import (
    assemble_matrix_block,
    create_vector_block,
    assemble_vector_block,
)
from meshing import create_divided_square


def u_e(x, module=np):
    "Function to represent the exact solution"
    # return module.exp(- ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / (2 * 0.15**2))
    return module.sin(module.pi * x[0]) * module.sin(module.pi * x[1])


# Set some parameters
num_time_steps = 10
k_0 = 3  # Polynomial degree in omega_0
k_1 = 3  # Polynomial degree in omgea_1
delta_t = 1  # TODO Make constant
h = 0.05  # Maximum cell diameter

# Create the mesh
comm = MPI.COMM_WORLD
msh, ct, ft, vol_ids, bound_ids = create_divided_square(comm, h)

# Create sub-meshes of omega_0 and omega_1 so that we can create
# different function spaces over each part of the domain
tdim = msh.topology.dim
submesh_0, sm_0_to_msh = mesh.create_submesh(msh, tdim, ct.find(vol_ids["omega_0"]))[:2]
submesh_1, sm_1_to_msh = mesh.create_submesh(msh, tdim, ct.find(vol_ids["omega_1"]))[:2]

# Define function spaces on each submesh
V_0 = fem.functionspace(submesh_0, ("Discontinuous Lagrange", k_0))
V_1 = fem.functionspace(submesh_1, ("Lagrange", k_1))
W = ufl.MixedFunctionSpace(V_0, V_1)

# Test and trial functions
u = ufl.TrialFunctions(W)
v = ufl.TestFunctions(W)

# We use msh as the integration domain, so we require maps from
# cells in msh to cells in submesh_0 and submesh_1
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
msh_to_sm_0 = np.full(num_cells, -1)
msh_to_sm_0[sm_0_to_msh] = np.arange(len(sm_0_to_msh))
msh_to_sm_1 = np.full(num_cells, -1)
msh_to_sm_1[sm_1_to_msh] = np.arange(len(sm_1_to_msh))

# Create integration entities for the interface integral
interface_facets = ft.find(bound_ids["interface"])
domain_0_cells = ct.find(vol_ids["omega_0"])
domain_1_cells = ct.find(vol_ids["omega_1"])
interface_entities, msh_to_sm_0, msh_to_sm_1 = interface_int_entities(
    msh, interface_facets, msh_to_sm_0, msh_to_sm_1
)

# Compute integration entities for boundary terms
boundary_entites = [
    (
        bound_ids["boundary_0"],
        fem.compute_integration_domains(
            fem.IntegralType.exterior_facet,
            msh.topology,
            ft.find(bound_ids["boundary_0"]),
            ft.dim,
        ),
    )
]

# Compute integration entities for the interior facet integrals
# over omega_0. These are needed for the DG scheme
omega_0_int_entities = interior_facet_int_entities(
    submesh_0, sm_0_to_msh
)

# Create measures
dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
ds = ufl.Measure("ds", domain=msh, subdomain_data=boundary_entites)
dS = ufl.Measure(
    "dS",
    domain=msh,
    subdomain_data=[
        (bound_ids["interface"], interface_entities),
        (bound_ids["omega_0_int_facets"], omega_0_int_entities),
    ],
)

# Define forms
# TODO Add k dependency
gamma_int = 10  # Penalty param on interface
gamma_dg = 10 * k_0**2  # Penalty parm for DG method
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

x = ufl.SpatialCoordinate(msh)
c = 1.0 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

u_0_n = fem.Function(V_0)
u_1_n = fem.Function(V_1)

w = ufl.as_vector((0.5 - x[1], 0.0))
lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), 1, 0)

# DG scheme in Omega_0
a = (
    inner(u[0] / delta_t, v[0]) * dx(vol_ids["omega_0"])
    - inner(w * u[0], grad(v[0])) * dx(vol_ids["omega_0"])
    + inner(
        lmbda("+") * dot(w("+"), n("+")) * u[0]("+")
        - lmbda("-") * dot(w("-"), n("-")) * u[0]("-"),
        jump(v[0]),
    )
    * dS(bound_ids["omega_0_int_facets"])
    + inner(lmbda * dot(w, n) * u[0], v[0]) * ds(bound_ids["boundary_0"])
    + inner(c * grad(u[0]), grad(v[0])) * dx(vol_ids["omega_0"])
    - inner(c * avg(grad(u[0])), jump(v[0], n)) * dS(bound_ids["omega_0_int_facets"])
    - inner(c * jump(u[0], n), avg(grad(v[0]))) * dS(bound_ids["omega_0_int_facets"])
    + (gamma_dg / avg(h))
    * inner(c * jump(u[0], n), jump(v[0], n))
    * dS(bound_ids["omega_0_int_facets"])
    - inner(c * grad(u[0]), v[0] * n) * ds(bound_ids["boundary_0"])
    - inner(c * grad(v[0]), u[0] * n) * ds(bound_ids["boundary_0"])
    + (gamma_dg / h) * inner(c * u[0], v[0]) * ds(bound_ids["boundary_0"])
)

# CG scheme in Omega_1
a += inner(u[1] / delta_t, v[1]) * dx(vol_ids["omega_1"]) + inner(
    c * grad(u[1]), grad(v[1])
) * dx(vol_ids["omega_1"])


# Coupling terms on the interface
def jump_i(v, n):
    return v[0]("+") * n("+") + v[1]("-") * n("-")


def grad_avg_i(v):
    return 1 / 2 * (grad(v[0]("+")) + grad(v[1]("-")))


a += (
    -inner(c * grad_avg_i(u), jump_i(v, n)) * dS(bound_ids["interface"])
    - inner(c * jump_i(u, n), grad_avg_i(v)) * dS(bound_ids["interface"])
    + gamma_int
    / avg(h)
    * inner(c * jump_i(u, n), jump_i(v, n))
    * dS(bound_ids["interface"])
)

# Compile LHS forms
entity_maps = {submesh_0: msh_to_sm_0, submesh_1: msh_to_sm_1}
a = fem.form(ufl.extract_blocks(a), entity_maps=entity_maps)

# Forms for the righ-hand side
f_0 = dot(w, grad(u_e(ufl.SpatialCoordinate(msh), module=ufl))) - div(
    c * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl))
)
f_1 = -div(c * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl)))

u_D = fem.Function(V_0)
u_D.interpolate(u_e)

L = (
    inner(f_0, v[0]) * dx(vol_ids["omega_0"])
    - inner((1 - lmbda) * dot(w, n) * u_D, v[0]) * ds(bound_ids["boundary_0"])
    + inner(u_0_n / delta_t, v[0]) * dx(vol_ids["omega_0"])
    - inner(c * u_D * n, grad(v[0])) * ds(bound_ids["boundary_0"])
    + gamma_dg / h * inner(c * u_D, v[0]) * ds(bound_ids["boundary_0"])
    + inner(f_1, v[1]) * dx(vol_ids["omega_1"])
    + inner(u_1_n / delta_t, v[1]) * dx(vol_ids["omega_1"])
)

# Compile RHS forms
L = fem.form(ufl.extract_blocks(L), entity_maps=entity_maps)

# Apply boundary condition. Since the boundary condition is applied on
# V_1, we must convert the facet tags to submesh_1 in order to locate
# the boundary degrees of freedom.
# NOTE: We don't do this for V_0 since the Dirichlet boundary condition
# is enforced weakly by the DG scheme.
ft_sm_1 = convert_facet_tags(msh, submesh_1, sm_1_to_msh, ft)
bound_facets_sm_1 = ft_sm_1.find(bound_ids["boundary_1"])
submesh_1.topology.create_connectivity(tdim - 1, tdim)
bound_dofs = fem.locate_dofs_topological(V_1, tdim - 1, bound_facets_sm_1)
u_bc_1 = fem.Function(V_1)
u_bc_1.interpolate(u_e)
bc_1 = fem.dirichletbc(u_bc_1, bound_dofs)
bcs = [bc_1]

# Assemble the system of equations
A = assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = create_vector_block(L)

# Set up solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")

# Setup files for visualisation
u_0_file = io.VTXWriter(msh.comm, "u_0.bp", [u_0_n._cpp_object], "BP4")
u_1_file = io.VTXWriter(msh.comm, "u_1.bp", [u_1_n._cpp_object], "BP4")

# Time stepping loop
t = 0.0
u_0_file.write(t)
u_1_file.write(t)
x = A.createVecRight()
for n in range(num_time_steps):
    t += delta_t

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector_block(b, L, a, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)

    # Recover solution
    offset = V_0.dofmap.index_map.size_local * V_0.dofmap.index_map_bs
    u_0_n.x.array[:offset] = x.array_r[:offset]
    u_1_n.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
    u_0_n.x.scatter_forward()
    u_1_n.x.scatter_forward()

    # Write to file
    u_0_file.write(t)
    u_1_file.write(t)

u_0_file.close()
u_1_file.close()

# Compute errors
e_L2_0 = norm_L2(msh.comm, u_0_n - u_e(ufl.SpatialCoordinate(submesh_0), module=ufl))
e_L2_1 = norm_L2(msh.comm, u_1_n - u_e(ufl.SpatialCoordinate(submesh_1), module=ufl))
e_L2 = np.sqrt(e_L2_0**2 + e_L2_1**2)

if msh.comm.rank == 0:
    print(f"e_L2 = {e_L2}")
