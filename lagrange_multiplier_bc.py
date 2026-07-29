# Solves u - div(grad(u)) = f, where the Dirichlet boundary condition is
# enforced via a Lagrange multiplier. See "The finite element method with
# Lagrangian multipliers" by Babuška (1973)

from mpi4py import MPI

import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import div, extract_blocks, grad, inner
from utils import norm_L2


# Manufactured solution. The exact Lagrange multiplier is its normal
# derivative, so a solution with a non-constant normal derivative is needed
# for the multiplier to be non-trivial
def u_exact(x):
    return ufl.sin(x[0]) * ufl.exp(x[1])


# Create mesh
l_x, l_y = 2.0, 1.0
n_x, n_y = 16, 8
msh = mesh.create_rectangle(comm=MPI.COMM_WORLD, points=((0.0, 0.0), (l_x, l_y)), n=(n_x, n_y))

# Create sub-mesh of the boundary to define function space for the Lagrange
# multiplier. The boundary condition applies to the whole boundary, so the
# exterior facets can be taken directly rather than located with a marker
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_entities(fdim)
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
submesh, submesh_emap = mesh.create_submesh(msh, fdim, boundary_facets)[0:2]

# Create function spaces on the mesh and sub-mesh. W should be the trace
# space of V on the boundary: a richer space (e.g. discontinuous P(k-1),
# which has the same dimension) leaves the constraint operator rank
# deficient, while a coarser one enforces the boundary condition too weakly
# and reduces the convergence rate of u
k = 3  # Polynomial degree
V = fem.functionspace(msh, ("Lagrange", k))
W = fem.functionspace(submesh, ("Lagrange", k))
X = ufl.MixedFunctionSpace(V, W)

# Trial and test functions
u, lmbda = ufl.TrialFunctions(X)
v, mu = ufl.TestFunctions(X)

# Exact solution and manufactured right-hand side
x = ufl.SpatialCoordinate(msh)
u_e = u_exact(x)
f = u_e - div(grad(u_e))

# Dirichlet boundary condition, interpolated into the Lagrange multiplier
# space on the boundary sub-mesh. The multiplier enforces the constraint
# weakly, pinning u to the L^2(boundary) projection of the data onto W.
# Interpolating first therefore makes u = u_d hold exactly (up to solver
# precision), whereas passing u_e directly would give its projection, which
# differs from its interpolant by O(h^(k+1))
u_d = fem.Function(W)
u_d.interpolate(
    fem.Expression(u_exact(ufl.SpatialCoordinate(submesh)), W.element.interpolation_points)
)

# Create integration measures. We take msh to be the integration domain
dx = ufl.Measure("dx", domain=msh)
ds = ufl.Measure("ds", domain=msh)

# Since our form involves multiple meshes, we need to provide maps relating
# the integration domain mesh (`msh`) to the other meshes in the form (just
# `submesh` here)
entity_maps = [submesh_emap]

# Define forms
a = inner(u, v) * dx + inner(grad(u), grad(v)) * dx - (inner(lmbda, v) * ds + inner(u, mu) * ds)
L = inner(f, v) * dx - inner(u_d, mu) * ds

# Extract block structure and create LinearProblem. We provide the entity maps here.
# The (2, 2) block of the saddle point system is zero, so the factorisation must
# pivot. PETSc's built-in LU fails with a missing diagonal entry, so use superlu_dist.
# MUMPS also works
uh, lmbdah = fem.Function(V), fem.Function(W)
petsc_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "superlu_dist"}
problem = LinearProblem(
    extract_blocks(a),
    extract_blocks(L),
    u=[uh, lmbdah],
    bcs=[],
    kind="mpi",
    petsc_options_prefix="lagrange_multiplier_bc_",
    petsc_options=petsc_opts,
    entity_maps=entity_maps,
)
problem.solve()

# Write to file
with io.VTXWriter(msh.comm, "u.bp", uh) as file:
    file.write(0.0)
with io.VTXWriter(msh.comm, "lmbda.bp", lmbdah) as file:
    file.write(0.0)

# Compute L^2-norm of error
e_L2 = norm_L2(msh.comm, uh - u_e)

# Compute the L^2-norm of the error between the interpolated boundary
# condition and the computed solution on the boundary. Since `u_d` is
# defined over `submesh` and the measure `ds` is defined over `msh`, the
# entity maps must be provided
e_L2_bdry = norm_L2(msh.comm, uh - u_d, measure=ds, entity_maps=entity_maps)

if msh.comm.rank == 0:
    print(f"e_L2 = {e_L2}")
    print(f"e_L2_bdry = {e_L2_bdry}")
