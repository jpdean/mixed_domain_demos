# In this demo, we implement a domain decomposition scheme for
# the Poisson equation based on Nitche's method. The scheme can
# be found in "Mortaring by a method of J. A. Nitsche" by Rolf
# Stenberg. See also "A finite element method for domain
# decomposition with non-matching grids" by Becker et al.

from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, avg, div
import numpy as np
from petsc4py import PETSc
from utils import norm_L2, convert_facet_tags, interface_int_entities
from dolfinx.fem.petsc import LinearProblem
from meshing import create_square_with_circle
from utils import jump_i, grad_avg_i


def u_e(x, module=np):
    "A function to represent the exact solution"
    return module.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / (2 * 0.05**2)) + x[0]


# Set some parameters
comm = MPI.COMM_WORLD
h = 0.05  # Maximum cell diameter
k_0 = 1  # Polynomial degree in omega_0
k_1 = 3  # Polynomial degree in omega_1

# Create mesh and sub-meshes
msh, ct, ft, vol_ids, surf_ids = create_square_with_circle(comm, h)
tdim = msh.topology.dim
domain_0_cells = ct.find(vol_ids["omega_0"])
domain_1_cells = ct.find(vol_ids["omega_1"])
submesh_0, sm_0_emap = mesh.create_submesh(msh, tdim, domain_0_cells)[:2]
submesh_1, sm_1_emap = mesh.create_submesh(msh, tdim, domain_1_cells)[:2]

# Define function spaces on each submesh
V_0 = fem.functionspace(submesh_0, ("Lagrange", k_0))
V_1 = fem.functionspace(submesh_1, ("Lagrange", k_1))
W = ufl.MixedFunctionSpace(V_0, V_1)

# Test and trial functions
u = ufl.TrialFunctions(W)
v = ufl.TestFunctions(W)

# We use msh as the integration domain, so we require maps relating cells
# in msh to cells in submesh_0 and submesh_1. These can be created
# as follows:
entity_maps = [sm_0_emap, sm_1_emap]

# Compute integration entities for the interface integral
fdim = tdim - 1
interface_facets = ft.find(surf_ids["interface"])

# Create interface integration entities. We provide a marker to identify which cells
# correspond to the "+" restriction and which correspond to the "-" restriction
marker = ct.values == vol_ids["omega_0"]
interface_entities = interface_int_entities(msh, interface_facets, marker)

# Create integration measures
dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
dS = ufl.Measure("dS", domain=msh, subdomain_data=[(surf_ids["interface"], interface_entities)])

x = ufl.SpatialCoordinate(msh)
kappa = [1.0 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) for _ in range(2)]

# Penalty parameter (including harmonic mean on kappa on interface)
# TODO Add k dependency
gamma = 10 * 2 * kappa[0] * kappa[1] / (kappa[0] + kappa[1])
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

a = (
    inner(kappa[0] * grad(u[0]), grad(v[0])) * dx(vol_ids["omega_0"])
    + inner(kappa[1] * grad(u[1]), grad(v[1])) * dx(vol_ids["omega_1"])
    - inner(grad_avg_i(u, kappa), jump_i(v, n)) * dS(surf_ids["interface"])
    - inner(jump_i(u, n), grad_avg_i(v, kappa)) * dS(surf_ids["interface"])
    + gamma / avg(h) * inner(jump_i(u, n), jump_i(v, n)) * dS(surf_ids["interface"])
)

# Define right-hand side forms
f = [-div(kap * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl))) for kap in kappa]
L = inner(f[0], v[0]) * dx(vol_ids["omega_0"]) + inner(f[1], v[1]) * dx(vol_ids["omega_1"])

# Apply boundary conditions. We require the DOFs of V_0 on the domain
# boundary. These can be identified via that facets of submesh_0 that
# lie on the domain boundary.
# FIXME Update function
smhs0_cell_imap = submesh_0.topology.index_map(tdim)
smsh_0_num_cells = smhs0_cell_imap.size_local + smhs0_cell_imap.num_ghosts
sm_0_to_msh = sm_0_emap.sub_topology_to_topology(
    np.arange(smsh_0_num_cells, dtype=np.int32), inverse=False
)
# print(sm_0_to_msh)
ft_sm_0 = convert_facet_tags(msh, submesh_0, sm_0_to_msh, ft)
bound_facets_sm_0 = ft_sm_0.find(surf_ids["boundary"])
submesh_0.topology.create_connectivity(fdim, tdim)
bound_dofs = fem.locate_dofs_topological(V_0, fdim, bound_facets_sm_0)
u_bc_0 = fem.Function(V_0)
u_bc_0.interpolate(u_e)
bc_0 = fem.dirichletbc(u_bc_0, bound_dofs)
bcs = [bc_0]

u_0, u_1 = fem.Function(V_0), fem.Function(V_1)
petsc_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "superlu_dist"}
problem = LinearProblem(
    ufl.extract_blocks(a),
    ufl.extract_blocks(L),
    u=[u_0, u_1],
    bcs=bcs,
    kind="mpi",
    petsc_options_prefix="poisson_domain_decomp_",
    petsc_options=petsc_opts,
    entity_maps=entity_maps,
)
problem.solve()

# Write solution to file
with io.VTXWriter(msh.comm, "u_0.bp", u_0, "BP4") as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "u_1.bp", u_1, "BP4") as f:
    f.write(0.0)

# Compute error in solution
e_L2_0 = norm_L2(msh.comm, u_0 - u_e(ufl.SpatialCoordinate(submesh_0), module=ufl))
e_L2_1 = norm_L2(msh.comm, u_1 - u_e(ufl.SpatialCoordinate(submesh_1), module=ufl))
e_L2 = np.sqrt(e_L2_0**2 + e_L2_1**2)

if msh.comm.rank == 0:
    print(e_L2)
