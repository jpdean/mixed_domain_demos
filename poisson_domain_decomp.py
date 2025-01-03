# In this demo, we implement a domain decomposition scheme for
# the Poisson equation based on Nitche's method. The scheme can
# be found in "Mortaring by a method of J. A. Nitsche" by Rolf
# Stenberg. See also "A finite element method for domain
# decomposition with non-matching grids" by Becker et al.

from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, avg, div
import numpy as np
from petsc4py import PETSc
from utils import norm_L2, convert_facet_tags, interface_int_entities
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from meshing import create_square_with_circle


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
submesh_0, sm_0_to_msh = mesh.create_submesh(msh, tdim, ct.find(vol_ids["omega_0"]))[:2]
submesh_1, sm_1_to_msh = mesh.create_submesh(msh, tdim, ct.find(vol_ids["omega_1"]))[:2]

# Define function spaces on each submesh
V_0 = fem.functionspace(submesh_0, ("Lagrange", k_0))
V_1 = fem.functionspace(submesh_1, ("Lagrange", k_1))
W = ufl.MixedFunctionSpace(V_0, V_1)

# Test and trial functions
u = ufl.TrialFunctions(W)
v = ufl.TestFunctions(W)

# We use msh as the integration domain, so we require maps from cells
# in msh to cells in submesh_0 and submesh_1. These can be created
# as follows:
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
msh_to_sm_0 = np.full(num_cells, -1)
msh_to_sm_0[sm_0_to_msh] = np.arange(len(sm_0_to_msh))
msh_to_sm_1 = np.full(num_cells, -1)
msh_to_sm_1[sm_1_to_msh] = np.arange(len(sm_1_to_msh))

# Compute integration entities for the interface integral
fdim = tdim - 1
interface_facets = ft.find(surf_ids["interface"])
domain_0_cells = ct.find(vol_ids["omega_0"])
domain_1_cells = ct.find(vol_ids["omega_1"])

# Create interface integration entities and modify msh_to_sm maps
interface_entities, msh_to_sm_0, msh_to_sm_1 = interface_int_entities(
    msh, interface_facets, msh_to_sm_0, msh_to_sm_1
)

# Create entity maps using the modified msh_to_sm maps
entity_maps = {submesh_0: msh_to_sm_0, submesh_1: msh_to_sm_1}

# Create integration measures
dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
dS = ufl.Measure(
    "dS", domain=msh, subdomain_data=[(surf_ids["interface"], interface_entities)]
)

# TODO Add k dependency
gamma = 10  # Penalty parameter
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

x = ufl.SpatialCoordinate(msh)
kappa = 1.0 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def jump_i(v, n):
    return v[0]("+") * n("+") + v[1]("-") * n("-")


def grad_avg_i(v):
    return 1 / 2 * (grad(v[0]("+")) + grad(v[1]("-")))


a = (
    inner(kappa * grad(u[0]), grad(v[0])) * dx(vol_ids["omega_0"])
    + inner(kappa * grad(u[1]), grad(v[1])) * dx(vol_ids["omega_1"])
    - inner(kappa * grad_avg_i(u), jump_i(v, n)) * dS(surf_ids["interface"])
    - inner(kappa * jump_i(u, n), grad_avg_i(v)) * dS(surf_ids["interface"])
    + gamma / avg(h) * inner(kappa * jump_i(u, n), jump_i(v, n)) * dS(surf_ids["interface"])
)

# Compile LHS forms
a = fem.form(ufl.extract_blocks(a), entity_maps=entity_maps)

# Define right-hand side forms
f = -div(kappa * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl)))
L = inner(f, v[0]) * dx(vol_ids["omega_0"]) + inner(f, v[1]) * dx(vol_ids["omega_1"])

# Compile RHS forms and set block structure
L = fem.form(ufl.extract_blocks(L), entity_maps=entity_maps)

# Apply boundary conditions. We require the DOFs of V_0 on the domain
# boundary. These can be identified via that facets of submesh_0 that
# lie on the domain boundary.
ft_sm_0 = convert_facet_tags(msh, submesh_0, sm_0_to_msh, ft)
bound_facets_sm_0 = ft_sm_0.find(surf_ids["boundary"])
submesh_0.topology.create_connectivity(fdim, tdim)
bound_dofs = fem.locate_dofs_topological(V_0, fdim, bound_facets_sm_0)
u_bc_0 = fem.Function(V_0)
u_bc_0.interpolate(u_e)
bc_0 = fem.dirichletbc(u_bc_0, bound_dofs)
bcs = [bc_0]

# Assemble linear system of equations
A = assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = assemble_vector_block(L, a, bcs=bcs)

# Set-up solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

# Recover solution
u_0, u_1 = fem.Function(V_0), fem.Function(V_1)
offset = V_0.dofmap.index_map.size_local * V_0.dofmap.index_map_bs
u_0.x.array[:offset] = x.array_r[:offset]
u_1.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
u_0.x.scatter_forward()
u_1.x.scatter_forward()

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
