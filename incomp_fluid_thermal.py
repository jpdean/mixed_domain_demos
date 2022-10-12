# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # A divergence conforming discontinuous Galerkin method for the Navier-Stokes equations
# This demo illustrates how to implement a divergence conforming
# discontinuous Galerkin method for the Navier-Stokes equations in
# FEniCSx. The method conserves mass exactly and uses upwinding. The
# formulation is based on a combination of "A fully divergence-free
# finite element method for magnetohydrodynamic equations" by Hiptmair
# et al., "A Note on Discontinuous Galerkin Divergence-free Solutions
# of the Navier-Stokes Equations" by Cockburn et al, and "On the Divergence
# Constraint in Mixed Finite Element Methods for Incompressible Flows" by
# John et al.

# ## Governing equations
# We consider the incompressible Navier-Stokes equations in a domain
# $\Omega \subset \mathbb{R}^d$, $d \in \{2, 3\}$, and time interval
# $(0, \infty)$, given by
# $$
#     \partial_t u - \nu \Delta u + (u \cdot \nabla)u + \nabla p = f
#     \textnormal{ in } \Omega_t,
# $$
# $$
#     \nabla \cdot u = 0
#     \textnormal{ in } \Omega_t,
# $$
# where $u: \Omega_t \to \mathbb{R}^d$ is the velocity field,
# $p: \Omega_t \to \mathbb{R}$ is the pressure field,
# $f: \Omega_t \to \mathbb{R}^d$ is a prescribed force, $\nu \in \mathbb{R}^+$
# is the kinematic viscosity, and
# $\Omega_t \coloneqq \Omega \times (0, \infty)$.

# The problem is supplemented with the initial condition
# $$
#     u(x, 0) = u_0(x) \textnormal{ in } \Omega
# $$
# and boundary condition
# $$
#     u = u_D \textnormal{ on } \partial \Omega \times (0, \infty),
# $$
# where $u_0: \Omega \to \mathbb{R}^d$ is a prescribed initial velocity field
# which satisfies the divergence free condition. The pressure field is only
# determined up to a constant, so we seek the unique pressure field satisfying
# $$
#     \int_\Omega p = 0.
# $$

# ## Discrete problem
# We begin by introducing the function spaces
# $$
#     V_h^g \coloneqq \left\{v \in H(\textnormal{div}; \Omega);
#     v|_K \in V_h(K) \; \forall K \in \mathcal{T}, v \cdot n = g \cdot n
#     \textnormal{ on } \partial \Omega \right\}
# $$,
# and
# $$
#     Q_h \coloneqq \left\{q \in L^2_0(\Omega);
#     q|_K \in Q_h(K) \; \forall K \in \mathcal{T} \right\}.
# $$
# The local spaces $V_h(K)$ and $Q_h(K)$ should satisfy
# $$
#     \nabla \cdot V_h(K) \subseteq Q_h(K),
# $$
# in order for mass to be conserved exactly. Suitable choices on
# affine simplex cells include
# $$
#     V_h(K) \coloneqq \mathbb{RT}_k(K) \textnormal{ and }
#     Q_h(K) \coloneqq \mathbb{P}_k(K),
# $$
# or
# $$
#     V_h(K) \coloneqq \mathbb{BDM}_k(K) \textnormal{ and }
#     Q_h(K) \coloneqq \mathbb{P}_{k-1}(K).
# $$

# Let two cells $K^+$ and $K^-$ share a facet $F$. The trace of a piecewise
# smooth vector valued function $\phi$ on F taken approaching from inside $K^+$
# (resp. $K^-$) is denoted $\phi^{+}$ (resp. $\phi^-$). We now introduce the
# average
# $\renewcommand{\avg}[1]{\left\{\!\!\left\{#1\right\}\!\!\right\}}$
# $$
#     \avg{\phi} = \frac{1}{2} \left(\phi^+ + \phi^-\right)
# $$
# $\renewcommand{\jump}[1]{\llbracket #1 \rrbracket}$
# and jump
# $$
#     \jump{\phi} = \phi^+ \otimes n^+ + \phi^- \otimes n^-,
# $$
# operators, where $n$ denotes the outward unit normal to $\partial K$.
# Finally, let the upwind flux of $\phi$ with respect to a vector field
# $\psi$ be defined as
# $$
#     \hat{\phi}^\psi \coloneqq
#     \begin{cases}
#         \lim_{\epsilon \downarrow 0} \phi(x - \epsilon \psi(x)), \;
#         x \in \partial K \setminus \Gamma^\psi, \\
#         0, \qquad \qquad \qquad \qquad x \in \partial K \cap \Gamma^\psi,
#     \end{cases}
# $$
# where $\Gamma^\psi = \left\{x \in \Gamma; \; \psi(x) \cdot n(x) < 0\right\}$.

# The semi-discrete version problem (in dimensionless form) is: find
# $(u_h, p_h) \in V_h^{u_D} \times Q_h$ such that
# $$
#     \int_\Omega \partial_t u_h \cdot v + a_h(u_h, v_h) + c_h(u_h; u_h, v_h)
#     + b_h(v_h, p_h) = \int_\Omega f \cdot v_h + L_{a_h}(v_h) + L_{c_h}(v_h)
#      \quad \forall v_h \in V_h^0,
# $$
# $$
#     b_h(u_h, q_h) = 0 \quad \forall q_h \in Q_h,
# $$
# where
# $\renewcommand{\sumK}[0]{\sum_{K \in \mathcal{T}_h}}$
# $\renewcommand{\sumF}[0]{\sum_{F \in \mathcal{F}_h}}$
# $$
#     a_h(u, v) = Re^{-1} \left(\sumK \int_K \nabla u : \nabla v
#     - \sumF \int_F \avg{\nabla u} : \jump{v}
#     - \sumF \int_F \avg{\nabla v} : \jump{u} \\
#     + \sumF \int_F \frac{\alpha}{h_K} \jump{u} : \jump{v}\right),
# $$
# $$
#     c_h(w; u, v) = - \sumK \int_K u \cdot \nabla \cdot (v \otimes w)
#     + \sumK \int_{\partial_K} w \cdot n \hat{u}^{w} \cdot v,
# $$
# $$
# L_{a_h}(v_h) = Re^{-1} \left(- \int_{\partial \Omega} u_D \otimes n :
#   \nabla_h v_h + \frac{\alpha}{h} u_D \otimes n : v_h \otimes n \right),
# $$
# $$
#     L_{c_h}(v_h) = - \int_{\partial \Omega} u_D \cdot n \hat{u}_D \cdot v_h,
# $$
# and
# $$
#     b_h(v, q) = - \int_K \nabla \cdot v q.
# $$

# ## Implementation
# We begin by importing the required modules and functions

from dolfinx import mesh, fem, io
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from ufl import (TrialFunction, TestFunction, CellDiameter, FacetNormal,
                 inner, grad, dx, dS, avg, outer, div, conditional,
                 gt, dot, Measure)
from ufl import jump
from utils import convert_facet_tags

# We also define some helper functions that will be used later


def norm_L2(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM))


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            fem.Constant(msh, PETSc.ScalarType(1.0)) * dx)), op=MPI.SUM)
    return 1 / vol * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * dx)), op=MPI.SUM)


# We define some simulation parameters

num_time_steps = 10
t_end = 0.1
R_e = 1000  # Reynolds Number
k = 2  # Polynomial degree

partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
msh, ct, ft = io.gmshio.read_from_msh(
    "benchmark.msh", MPI.COMM_WORLD, gdim=2, partitioner=partitioner)
msh.name = "benchmark"
ct.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_facets"

volume_id = {"fluid": 6,
             "solid": 7}
boundary_id = {"inlet": 8,
               "outlet": 9,
               "wall": 10,
               "obstacle": 11}

# Create fluid submesh
fluid_cells = ct.indices[ct.values == volume_id["fluid"]]
tdim = msh.topology.dim
fluid_submesh, fluid_entity_map = mesh.create_submesh(
    msh, tdim, fluid_cells)[:2]

# Convert meshtags for use on fluid_submesh
fluid_submesh_ft = convert_facet_tags(msh, fluid_submesh, fluid_entity_map, ft)

# Create submesh for cylinder
solid_cells = ct.indices[ct.values == volume_id["solid"]]
solid_submesh, solid_entity_map = mesh.create_submesh(
    msh, tdim, solid_cells)[:2]

# Function space for the velocity
V = fem.FunctionSpace(fluid_submesh, ("Raviart-Thomas", k + 1))
# Function space for the pressure
Q = fem.FunctionSpace(fluid_submesh, ("Discontinuous Lagrange", k))
# Funcion space for visualising the velocity field
W = fem.VectorFunctionSpace(fluid_submesh, ("Discontinuous Lagrange", k + 1))
# Function space for the solid domain
X = fem.FunctionSpace(solid_submesh, ("Lagrange", k))

# Define trial and test functions

# Velocity
u, v = TrialFunction(V), TestFunction(V)
# Pressure
p, q = TrialFunction(Q), TestFunction(Q)
# Fluid temperature
T_f, w_f = TrialFunction(Q), TestFunction(Q)
# Solid temperature
T_s, w_s = TrialFunction(X), TestFunction(X)

# Define some constants
delta_t = fem.Constant(fluid_submesh, PETSc.ScalarType(t_end / num_time_steps))
alpha = fem.Constant(fluid_submesh, PETSc.ScalarType(6.0 * k**2))
alpha_T = fem.Constant(fluid_submesh, PETSc.ScalarType(10.0 * k**2))
R_e_const = fem.Constant(fluid_submesh, PETSc.ScalarType(R_e))
kappa = fem.Constant(fluid_submesh, PETSc.ScalarType(0.01))

# List of tuples of form (id, expression)
dirichlet_bcs = [
    (boundary_id["inlet"], lambda x: np.vstack(
        ((1.5 * 4 * x[1] * (0.41 - x[1])) / 0.41**2, np.zeros_like(x[0])))),
    (boundary_id["wall"], lambda x: np.vstack(
        (np.zeros_like(x[0]), np.zeros_like(x[0])))),
    (boundary_id["obstacle"], lambda x: np.vstack(
        (np.zeros_like(x[0]), np.zeros_like(x[0]))))]
neumann_bcs = [(boundary_id["outlet"], fem.Constant(
    fluid_submesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType)))]

ds = Measure("ds", domain=fluid_submesh, subdomain_data=fluid_submesh_ft)

h = CellDiameter(fluid_submesh)
n = FacetNormal(fluid_submesh)


def jump_ns(phi, n):
    return outer(phi("+"), n("+")) + outer(phi("-"), n("-"))


# We solve the Stokes problem for the initial condition, so the convective
# terms are omitted for now

a_00 = 1 / R_e_const * (inner(grad(u), grad(v)) * dx
                        - inner(avg(grad(u)), jump_ns(v, n)) * dS
                        - inner(jump_ns(u, n), avg(grad(v))) * dS
                        + alpha / avg(h) * inner(jump_ns(u, n), jump_ns(v, n)) * dS)
a_01 = - inner(p, div(v)) * dx
a_10 = - inner(div(u), q) * dx
a_11 = fem.Constant(fluid_submesh, PETSc.ScalarType(0.0)) * inner(p, q) * dx

f = fem.Function(W)
# NOTE: Arrived at Neumann BC term by rewriting inner(grad(u), outer(v, n))
# it is based on as inner(dot(grad(u), n), v) and then g = dot(grad(u), n)
# etc. TODO Check this. NOTE Consider changing formulation to one with momentum
# law in conservative form to be able to specify momentum flux
L_0 = inner(f, v) * dx
L_1 = inner(fem.Constant(fluid_submesh, PETSc.ScalarType(0.0)), q) * dx

bcs = []
for bc in dirichlet_bcs:
    a_00 += 1 / R_e_const * (- inner(grad(u), outer(v, n)) * ds(bc[0])
                             - inner(outer(u, n), grad(v)) * ds(bc[0])
                             + alpha / h * inner(outer(u, n), outer(v, n)) * ds(bc[0]))
    u_D = fem.Function(V)
    u_D.interpolate(bc[1])
    L_0 += 1 / R_e_const * (- inner(outer(u_D, n), grad(v)) * ds(bc[0])
                            + alpha / h * inner(outer(u_D, n), outer(v, n)) * ds(bc[0]))

    bc_boundary_facets = fluid_submesh_ft.indices[fluid_submesh_ft.values == bc[0]]
    bc_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, bc_boundary_facets)
    bcs.append(fem.dirichletbc(u_D, bc_dofs))


for bc in neumann_bcs:
    L_0 += 1 / R_e_const * inner(bc[1], v) * ds(bc[0])

a = fem.form([[a_00, a_01],
              [a_10, a_11]])
L = fem.form([L_0,
              L_1])

# If the pressure is only determined up to a constant, pin a single degree
# of freedom

# TODO TIDY
# FIXME This assumes there is a vertex at point (0, 0)
if len(neumann_bcs) == 0:
    pressure_dofs = fem.locate_dofs_geometrical(
        Q, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                    np.isclose(x[1], 0.0)))
    if len(pressure_dofs) > 0:
        pressure_dof = [pressure_dofs[0]]
    else:
        pressure_dof = []
    bc_p = fem.dirichletbc(PETSc.ScalarType(0.0),
                           np.array(pressure_dof, dtype=np.int32),
                           Q)

# Assemble Stokes problem

A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

# Create and configure solver

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()
# See https://graal.ens-lyon.fr/MUMPS/doc/userguide_5.5.1.pdf
# TODO Check
opts["mat_mumps_icntl_6"] = 2
opts["mat_mumps_icntl_14"] = 100
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

# Solve Stokes for initial condition

x = A.createVecRight()
ksp.solve(b, x)

# Split the solution

u_h = fem.Function(V)
p_h = fem.Function(Q)
p_h.name = "p"
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u_h.x.array[:offset] = x.array_r[:offset]
u_h.x.scatter_forward()
p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
p_h.x.scatter_forward()

u_vis = fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)

# Write initial condition to file

u_file = io.VTXWriter(msh.comm, "u.bp", [u_vis._cpp_object])
p_file = io.VTXWriter(msh.comm, "p.bp", [p_h._cpp_object])
t = 0.0
u_file.write(t)
p_file.write(t)

# Create function to store solution and previous time step

u_n = fem.Function(V)
u_n.x.array[:] = u_h.x.array

# Solid heat diffusion problem
h_f = 100.0  # Heat transfer coeff
delta_t_T_s = fem.Constant(
    solid_submesh, PETSc.ScalarType(t_end / num_time_steps))
kappa_T_s = fem.Constant(solid_submesh, PETSc.ScalarType(0.1))
a_T_s = fem.form(inner(T_s / delta_t_T_s, w_s) * dx
                 + kappa_T_s * inner(grad(T_s), grad(w_s)) * dx
                 + kappa_T_s * inner(h_f * T_s, w_s) * ufl.ds)
# Solid temperature at previous time step
T_s_n = fem.Function(X)
L_T_s = fem.form(inner(T_s_n / delta_t_T_s + 1.0, w_s) * dx)

A_T_s = fem.petsc.assemble_matrix(a_T_s)
A_T_s.assemble()
b_T_s = fem.petsc.create_vector(L_T_s)

# Fluid heat convection-diffusion problem

dirichlet_bcs_T_f = [(boundary_id["inlet"], lambda x: np.zeros_like(x[0]))]
neumann_bcs_T_f = [(boundary_id["outlet"], lambda x: np.zeros_like(x[0]))]
robin_bcs_T_f = [(boundary_id["wall"], (0.0, 0.0))]

lmbda = conditional(gt(dot(u_n, n), 0), 1, 0)
a_T_f = inner(T_f / delta_t, w_f) * dx - \
    inner(u_h * T_f, grad(w_f)) * dx + \
    inner(lmbda("+") * dot(u_h("+"), n("+")) * T_f("+") -
          lmbda("-") * dot(u_h("-"), n("-")) * T_f("-"), jump(w_f)) * dS + \
    inner(lmbda * dot(u_h, n) * T_f, w_f) * ds + \
    kappa * (inner(grad(T_f), grad(w_f)) * dx -
             inner(avg(grad(T_f)), jump(w_f, n)) * dS -
             inner(jump(T_f, n), avg(grad(w_f))) * dS +
             (alpha / avg(h)) * inner(jump(T_f, n), jump(w_f, n)) * dS)

# Fluid temp at previous time step
T_f_n = fem.Function(Q)
L_T_f = inner(T_f_n / delta_t, w_f) * dx

for bc in dirichlet_bcs_T_f:
    T_D = fem.Function(Q)
    T_D.interpolate(bc[1])
    a_T_f += kappa * (- inner(grad(T_f), w_f * n) * ds(bc[0]) -
                      inner(grad(w_f), T_f * n) * ds(bc[0]) +
                      (alpha / h) * inner(T_f, w_f) * ds(bc[0]))
    L_T_f += - inner((1 - lmbda) * dot(u_h, n) * T_D, w_f) * ds(bc[0]) + \
        kappa * (- inner(T_D * n, grad(w_f)) * ds(bc[0]) +
                 (alpha / h) * inner(T_D, w_f) * ds(bc[0]))

for bc in neumann_bcs_T_f:
    g_T = fem.Function(Q)
    g_T.interpolate(bc[1])
    L_T_f += kappa * inner(g_T, w_f) * ds(bc[0])

for bc in robin_bcs_T_f:
    alpha_R, beta_R = bc[1]
    a_T_f += kappa * inner(alpha_R * T_f, w_f) * ds(bc[0])
    L_T_f += kappa * inner(beta_R, w_f) * ds(bc[0])

# Obstacle
a_T_f += kappa * inner(h_f * T_f, w_f) * ds(boundary_id["obstacle"])

obstacle_facets = ft.indices[ft.values == boundary_id["obstacle"]]
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
entity_maps = {fluid_submesh: [fluid_entity_map.index(entity)
                               if entity in fluid_entity_map else -1
                               for entity in range(num_cells)],
               solid_submesh: [solid_entity_map.index(entity)
                               if entity in solid_entity_map else -1
                               for entity in range(num_cells)]}

facet_integration_entities = {1: []}
fdim = tdim - 1
facet_imap = msh.topology.index_map(fdim)
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
for facet in obstacle_facets:
    # Check if this facet is owned
    if facet < facet_imap.size_local:
        cells = f_to_c.links(facet)
        assert len(cells) == 2

        cell_plus = cells[0] if cells[0] in fluid_cells else cells[1]
        cell_minus = cells[0] if cells[0] in solid_cells else cells[1]
        assert cell_plus in fluid_cells
        assert cell_minus in solid_cells

        # FIXME Don't use tolist
        local_facet_plus = c_to_f.links(cell_plus).tolist().index(facet)
        local_facet_minus = c_to_f.links(cell_minus).tolist().index(facet)
        facet_integration_entities[1].extend(
            [cell_plus, local_facet_plus, cell_minus, local_facet_minus])

        # HACK See test_assemble_submesh.py::test_jørgen_problem
        entity_maps[fluid_submesh][cell_minus] = \
            entity_maps[fluid_submesh][cell_plus]
        # Same hack for the right submesh
        entity_maps[solid_submesh][cell_plus] = \
            entity_maps[solid_submesh][cell_minus]
dS_coupling = Measure(
    "dS", domain=msh, subdomain_data=facet_integration_entities)

# TODO Add code to suport multiple domains in a single form
L_T_f_coupling = kappa * inner(h_f * T_s_n("-"), w_f("+")) * dS_coupling(1)

a_T_f = fem.form(a_T_f)
L_T_f = fem.form(L_T_f)
L_T_f_coupling = fem.form(L_T_f_coupling, entity_maps=entity_maps)

L_T_s_coupling = kappa_T_s * inner(h_f * T_f_n("+"), w_s("-")) * dS_coupling(1)
L_T_s_coupling = fem.form(L_T_s_coupling, entity_maps=entity_maps)

A_T_f = fem.petsc.create_matrix(a_T_f)
b_T_f = fem.petsc.create_vector(L_T_f)

ksp_T_f = PETSc.KSP().create(msh.comm)
ksp_T_f.setOperators(A_T_f)
ksp_T_f.setType("preonly")
ksp_T_f.getPC().setType("lu")
ksp_T_f.getPC().setFactorSolverType("superlu_dist")

ksp_T_s = PETSc.KSP().create(msh.comm)
ksp_T_s.setOperators(A_T_s)
ksp_T_s.setType("preonly")
ksp_T_s.getPC().setType("lu")
ksp_T_s.getPC().setFactorSolverType("superlu_dist")

T_f_file = io.VTXWriter(msh.comm, "T_f.bp", [T_f_n._cpp_object])
T_f_file.write(t)

# FIXME Why won't T_s.bp output upen when written in parallel?
# T_s_file = io.VTXWriter(msh.comm, "T_s.bp", [T_s_n._cpp_object])
# T_s_file.write(t)
T_s_file = io.XDMFFile(msh.comm, "T_s.xdmf", "w")
T_s_file.write_mesh(solid_submesh)
T_s_file.write_function(T_s_n, t)

# Now we add the time stepping and convective terms

u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")
a_00 += inner(u / delta_t, v) * dx - \
    inner(u, div(outer(v, u_n))) * dx + \
    inner((dot(u_n, n))("+") * u_uw, v("+")) * dS + \
    inner((dot(u_n, n))("-") * u_uw, v("-")) * dS + \
    inner(dot(u_n, n) * lmbda * u, v) * ds
a = fem.form([[a_00, a_01],
              [a_10, a_11]])

L_0 += inner(u_n / delta_t, v) * dx

for bc in dirichlet_bcs:
    u_D = fem.Function(V)
    u_D.interpolate(bc[1])
    L_0 += - inner(dot(u_n, n) * (1 - lmbda) * u_D, v) * ds(bc[0])
L = fem.form([L_0,
              L_1])

# Time stepping loop

for n in range(num_time_steps):
    t += delta_t.value

    A.zeroEntries()
    fem.petsc.assemble_matrix_block(A, a, bcs=bcs)
    A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)
    u_h.x.array[:offset] = x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()

    A_T_f.zeroEntries()
    fem.petsc.assemble_matrix(A_T_f, a_T_f)
    A_T_f.assemble()

    with b_T_f.localForm() as b_T_loc:
        b_T_loc.set(0)
    fem.petsc.assemble_vector(b_T_f, L_T_f)
    fem.petsc.assemble_vector(b_T_f, L_T_f_coupling)
    b_T_f.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

    ksp_T_f.solve(b_T_f, T_f_n.vector)
    T_f_n.x.scatter_forward()

    with b_T_s.localForm() as b_T_s_loc:
        b_T_s_loc.set(0)
    fem.petsc.assemble_vector(b_T_s, L_T_s)
    fem.petsc.assemble_vector(b_T_s, L_T_s_coupling)
    b_T_s.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp_T_s.solve(b_T_s, T_s_n.vector)
    T_s_n.x.scatter_forward()

    u_vis.interpolate(u_h)

    # Write to file
    u_file.write(t)
    p_file.write(t)
    T_f_file.write(t)
    T_s_file.write_function(T_s_n, t)

    # Update u_n
    u_n.x.array[:] = u_h.x.array

u_file.close()
p_file.close()
T_f_file.close()
T_s_file.close()

# Compute errors
e_div_u = norm_L2(msh.comm, div(u_h))
# This scheme conserves mass exactly, so check this
assert np.isclose(e_div_u, 0.0)
