# TODO This demo needs tidying and simplifying

from ufl import SpatialCoordinate
from hdg_navier_stokes import SolverType, Scheme, set_up, BCType
from dolfinx import fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from ufl import (TrialFunction, TestFunction, CellDiameter, FacetNormal,
                 inner, grad, dx, dS, avg, outer, div, conditional,
                 gt, dot, Measure, as_vector)
from ufl import jump as jump_T
import gmsh
from utils import convert_facet_tags
import sys


def generate_mesh(comm, h=0.1, h_fac=1/3):
    gmsh.initialize()

    volume_id = {"fluid": 1,
                 "solid": 2}

    boundary_id = {"left": 2,
                   "right": 3,
                   "bottom": 4,
                   "top": 5,
                   "obstacle": 6}
    gdim = 2

    if comm.rank == 0:

        gmsh.model.add("model")
        factory = gmsh.model.geo

        length = 1
        height = 2
        c = (0.49, 0.1)

        o_w = 0.075
        o_h = 0.02

        rectangle_points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(length, 0.0, 0.0, h),
            factory.addPoint(length, height, 0.0, h),
            factory.addPoint(0.0, height, 0.0, h)
        ]

        centre_point = factory.addPoint(c[0], c[1], 0.0, h * h_fac)
        obstacle_points = [
            factory.addPoint(c[0] + o_w / 2, c[1], 0.0, h * h_fac),
            factory.addPoint(c[0], c[1] - o_w / 2, 0.0, h * h_fac),
            factory.addPoint(c[0] - o_w / 2, c[1], 0.0, h * h_fac),
            factory.addPoint(c[0] - o_w / 2, c[1] + o_h, 0.0, h * h_fac),
            factory.addPoint(c[0] + o_w / 2, c[1] + o_h, 0.0, h * h_fac)
        ]

        rectangle_lines = [
            factory.addLine(rectangle_points[0], rectangle_points[1]),
            factory.addLine(rectangle_points[1], rectangle_points[2]),
            factory.addLine(rectangle_points[2], rectangle_points[3]),
            factory.addLine(rectangle_points[3], rectangle_points[0])
        ]

        obstacle_lines = [
            factory.addCircleArc(
                obstacle_points[0], centre_point, obstacle_points[1]),
            factory.addCircleArc(
                obstacle_points[1], centre_point, obstacle_points[2]),
            factory.addLine(obstacle_points[2], obstacle_points[3]),
            factory.addLine(obstacle_points[3], obstacle_points[4]),
            factory.addLine(obstacle_points[4], obstacle_points[0]),
        ]

        rectangle_curve = factory.addCurveLoop(rectangle_lines)
        circle_curve = factory.addCurveLoop(obstacle_lines)

        square_surface = factory.addPlaneSurface(
            [rectangle_curve, circle_curve])
        circle_surface = factory.addPlaneSurface([circle_curve])

        factory.synchronize()

        gmsh.model.addPhysicalGroup(2, [square_surface], volume_id["fluid"])
        gmsh.model.addPhysicalGroup(2, [circle_surface], volume_id["solid"])

        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[0]], boundary_id["bottom"])
        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[1]], boundary_id["right"])
        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[2]], boundary_id["top"])
        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[3]], boundary_id["left"])
        gmsh.model.addPhysicalGroup(1, obstacle_lines, boundary_id["obstacle"])

        gmsh.model.mesh.generate(2)

        # gmsh.fltk.run()

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, ct, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
    ft.name = "Facet markers"

    return msh, ct, ft, volume_id, boundary_id


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


def zero(x): return np.vstack(
    (np.zeros_like(x[0]),
     np.zeros_like(x[0])))


def par_print(string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


# We define some simulation parameters
num_time_steps = 10
t_end = 0.1
h = 0.05
nu = 1e-3
h_fac = 1 / 3  # Factor scaling h near the cylinder
k = 2  # Polynomial degree
solver_type = SolverType.NAVIER_STOKES
g = as_vector((0.0, -9.81))
rho_0 = 1.0  # Reference density (buoyancy term)
eps = 10.0  # Thermal expansion coefficient
f_T = 100.0  # Thermal source
gamma_int = 10  # Penalty param for temperature on interface
kappa = 0.001  # Thermal conductivity
alpha = 6.0 * k**2  # Penalty param for DG temp solver
# TODO Add other params

# Create mesh
comm = MPI.COMM_WORLD
msh, ct, ft, volume_id, boundary_id = generate_mesh(comm, h=h, h_fac=h_fac)

# Create submeshes of fluid and solid domains
tdim = msh.topology.dim
submesh_f, entity_map_f = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == volume_id["fluid"]])[:2]
submesh_s, entity_map_s = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == volume_id["solid"]])[:2]

# Convert meshtags to fluid submesh
fdim = tdim - 1
submesh_f.topology.create_connectivity(fdim, tdim)
ft_f = convert_facet_tags(msh, submesh_f, entity_map_f, ft)

# Define boundary conditions for fluid solver
boundary_conditions = {"bottom": (BCType.Dirichlet, zero),
                       "right": (BCType.Dirichlet, zero),
                       "top": (BCType.Dirichlet, zero),
                       "left": (BCType.Dirichlet, zero),
                       "obstacle": (BCType.Dirichlet, zero)}

# Function spaces for fluid and solid temperature
Q = fem.FunctionSpace(submesh_f, ("Discontinuous Lagrange", k))
Q_s = fem.FunctionSpace(submesh_s, ("Lagrange", k))
# Fluid and solid temperature at previous time step
T_n = fem.Function(Q)
T_s_n = fem.Function(Q_s)

# Time step
delta_t = t_end / num_time_steps  # TODO Make constant
# Buoyancy force
# TODO Figure out correct way of "linearising"
# For buoyancy term, see
# https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy)
# where I've omitted the rho g h part (can think of this is
# lumping gravity in with pressure, see 2P4 notes) and taken
# T_0 to be 0
rho_0 = fem.Constant(submesh_f, PETSc.ScalarType(rho_0))
eps = fem.Constant(submesh_f, PETSc.ScalarType(eps))
# Buoyancy force
f = - eps * rho_0 * T_n * g

# Set up fluid solver
(A, a, L, u_vis, p_h, ubar_n, pbar_h, u_offset, p_offset, ubar_offset,
 bcs, u_n, facet_mesh) = set_up(
    submesh_f, Scheme.DRW, ft_f, k, solver_type, boundary_conditions,
    boundary_id, f, delta_t, nu)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()
opts["mat_mumps_icntl_6"] = 2
opts["mat_mumps_icntl_14"] = 100
ksp.setFromOptions()

b = fem.petsc.create_vector_block(L)
x = A.createVecRight()

# Trial and test function for fluid temperature
T, w = TrialFunction(Q), TestFunction(Q)
# Trial and test funcitons for the solid temperature
T_s, w_s = TrialFunction(Q_s), TestFunction(Q_s)

# Convert to Constants
delta_t = fem.Constant(msh, PETSc.ScalarType(delta_t))
alpha = fem.Constant(msh, PETSc.ScalarType(alpha))
kappa = fem.Constant(msh, PETSc.ScalarType(kappa))

# Boundary conditions for the thermal solver
dirichlet_bcs_T = [(boundary_id["bottom"], lambda x: np.zeros_like(x[0])),
                   (boundary_id["right"], lambda x: np.zeros_like(x[0])),
                   (boundary_id["top"], lambda x: np.zeros_like(x[0])),
                   (boundary_id["left"], lambda x: np.zeros_like(x[0]))]

# Create entity maps
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
inv_entity_map_f = np.full(num_cells, -1)
inv_entity_map_f[entity_map_f] = np.arange(len(entity_map_f))
inv_entity_map_s = np.full(num_cells, -1)
inv_entity_map_s[entity_map_s] = np.arange(len(entity_map_s))
entity_maps = {submesh_f: inv_entity_map_f,
               submesh_s: inv_entity_map_s}

dx_T = Measure("dx", domain=msh, subdomain_data=ct)
ds_T = Measure("ds", domain=msh, subdomain_data=ft)

# Create measure for integration. Assign the first (cell, local facet)
# pair to the cell in omega_0, corresponding to the "+" restriction. Assign
# the second pair to the omega_1 cell, corresponding to the "-" restriction.
fluid_int_facets = 7  # FIXME Don't hardcode
# facet_integration_entities = {boundary_id["obstacle"]: [],
#                               fluid_int_facets: []}
obstacle_facet_entities = []
facet_imap = msh.topology.index_map(fdim)
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
domain_f_cells = ct.indices[ct.values == volume_id["fluid"]]
domain_s_cells = ct.indices[ct.values == volume_id["solid"]]
interface_facets = ft.indices[ft.values == boundary_id["obstacle"]]
for facet in interface_facets:
    # Check if this facet is owned
    if facet < facet_imap.size_local:
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        cell_plus = cells[0] if cells[0] in domain_f_cells else cells[1]
        cell_minus = cells[0] if cells[0] in domain_s_cells else cells[1]
        assert cell_plus in domain_f_cells
        assert cell_minus in domain_s_cells

        # FIXME Don't use tolist
        local_facet_plus = c_to_f.links(
            cell_plus).tolist().index(facet)
        local_facet_minus = c_to_f.links(
            cell_minus).tolist().index(facet)
        obstacle_facet_entities.extend(
            [cell_plus, local_facet_plus, cell_minus, local_facet_minus])

        # HACK cell_minus does not exist in the left submesh, so it will
        # be mapped to index -1. This is problematic for the assembler,
        # which assumes it is possible to get the full macro dofmap for the
        # trial and test functions, despite the restriction meaning we
        # don't need the non-existant dofs. To fix this, we just map
        # cell_minus to the cell corresponding to cell plus. This will
        # just add zeros to the assembled system, since there are no
        # u("-") terms. Could map this to any cell in the submesh, but
        # I think using the cell on the other side of the facet means a
        # facet space coefficient could be used
        entity_maps[submesh_f][cell_minus] = \
            entity_maps[submesh_f][cell_plus]
        # Same hack for the right submesh
        entity_maps[submesh_s][cell_plus] = \
            entity_maps[submesh_s][cell_minus]

# FIXME Do this more efficiently
submesh_f.topology.create_entities(fdim)
submesh_f.topology.create_connectivity(tdim, fdim)
submesh_f.topology.create_connectivity(fdim, tdim)
c_to_f_submesh_f = submesh_f.topology.connectivity(tdim, fdim)
f_to_c_submesh_f = submesh_f.topology.connectivity(fdim, tdim)
fluid_int_facet_entities = []
for facet in range(submesh_f.topology.index_map(fdim).size_local):
    cells = f_to_c_submesh_f.links(facet)
    if len(cells) == 2:
        # FIXME Don't use tolist
        local_facet_plus = c_to_f_submesh_f.links(
            cells[0]).tolist().index(facet)
        local_facet_minus = c_to_f_submesh_f.links(
            cells[1]).tolist().index(facet)

        fluid_int_facet_entities.extend(
            [entity_map_f[cells[0]], local_facet_plus,
             entity_map_f[cells[1]], local_facet_minus])
facet_integration_entities = [
    (boundary_id["obstacle"], obstacle_facet_entities),
    (fluid_int_facets, fluid_int_facet_entities)]
dS_T = Measure("dS", domain=msh,
               subdomain_data=facet_integration_entities)

h_T = CellDiameter(msh)
n_T = FacetNormal(msh)
lmbda_T = conditional(gt(dot(u_n, n_T), 0), 1, 0)

# Fluid velocity at current time step
u_h = u_n.copy()

# Define forms for the thermal problem
a_T_00 = inner(T / delta_t, w) * dx_T(volume_id["fluid"]) - \
    inner(u_h * T, grad(w)) * dx_T(volume_id["fluid"]) + \
    inner(lmbda_T("+") * dot(u_h("+"), n_T("+")) * T("+") -
          lmbda_T("-") * dot(u_h("-"), n_T("-")) * T("-"),
          jump_T(w)) * dS_T(fluid_int_facets) + \
    inner(lmbda_T * dot(u_h, n_T) * T, w) * ds_T + \
    kappa * (inner(grad(T), grad(w)) * dx_T(volume_id["fluid"]) -
             inner(avg(grad(T)), jump_T(w, n_T)) * dS_T(fluid_int_facets) -
             inner(jump_T(T, n_T), avg(grad(w))) * dS_T(fluid_int_facets) +
             (alpha / avg(h_T)) * inner(
        jump_T(T, n_T), jump_T(w, n_T)) * dS_T(fluid_int_facets)) \
    + kappa * (gamma_int / avg(h_T) * inner(
        T("+"), w("+")) * dS_T(boundary_id["obstacle"])
    - inner(1 / 2 * dot(grad(T("+")), n_T("+")),
            w("+")) * dS_T(boundary_id["obstacle"])
    - inner(1 / 2 * dot(grad(w("+")), n_T("+")),
            T("+")) * dS_T(boundary_id["obstacle"]))

a_T_01 = kappa * (- gamma_int / avg(h_T) * inner(
    T_s("-"), w("+")) * dS_T(boundary_id["obstacle"])
    + inner(1 / 2 * dot(grad(T_s("-")), n_T("-")),
            w("+")) * dS_T(boundary_id["obstacle"])
    + inner(1 / 2 * dot(grad(w("+")), n_T("+")),
            T_s("-")) * dS_T(boundary_id["obstacle"]))

a_T_10 = kappa * (- gamma_int / avg(h_T) * inner(
    T("+"), w_s("-")) * dS_T(boundary_id["obstacle"])
    + inner(1 / 2 * dot(grad(T("+")), n_T("+")),
            w_s("-")) * dS_T(boundary_id["obstacle"])
    + inner(1 / 2 * dot(grad(w_s("-")), n_T("-")),
            T("+")) * dS_T(boundary_id["obstacle"]))

a_T_11 = inner(T_s / delta_t, w_s) * dx_T(volume_id["solid"]) \
    + kappa * (inner(grad(T_s), grad(w_s)) * dx_T(volume_id["solid"])
               + gamma_int / avg(h_T) * inner(
        T_s("-"), w_s("-")) * dS_T(boundary_id["obstacle"])
    - inner(1 / 2 * dot(grad(T_s("-")), n_T("-")),
            w_s("-")) * dS_T(boundary_id["obstacle"])
    - inner(1 / 2 * dot(grad(w_s("-")), n_T("-")),
            T_s("-")) * dS_T(boundary_id["obstacle"]))

L_T_0 = inner(T_n / delta_t, w) * dx_T(volume_id["fluid"])

for bc in dirichlet_bcs_T:
    T_D = fem.Function(Q)
    T_D.interpolate(bc[1])
    a_T_00 += kappa * (- inner(grad(T), w * n_T) * ds_T(bc[0]) -
                       inner(grad(w), T * n_T) * ds_T(bc[0]) +
                       (alpha / h_T) * inner(T, w) * ds_T(bc[0]))
    L_T_0 += - inner((1 - lmbda_T) * dot(u_h, n_T) * T_D, w) * ds_T(bc[0]) + \
        kappa * (- inner(T_D * n_T, grad(w)) * ds_T(bc[0]) +
                 (alpha / h_T) * inner(T_D, w) * ds_T(bc[0]))

L_T_1 = inner(f_T, w_s) * dx_T(volume_id["solid"]) \
    + inner(T_s_n / delta_t, w_s) * dx_T(volume_id["solid"])

a_T_00 = fem.form(a_T_00, entity_maps=entity_maps)
a_T_01 = fem.form(a_T_01, entity_maps=entity_maps)
a_T_10 = fem.form(a_T_10, entity_maps=entity_maps)
a_T_11 = fem.form(a_T_11, entity_maps=entity_maps)

L_T_0 = fem.form(L_T_0, entity_maps=entity_maps)
L_T_1 = fem.form(L_T_1, entity_maps=entity_maps)

a_T = [[a_T_00, a_T_01],
       [a_T_10, a_T_11]]
L_T = [L_T_0, L_T_1]

A_T = fem.petsc.create_matrix_block(a_T)
b_T = fem.petsc.create_vector_block(L_T)

ksp_T = PETSc.KSP().create(msh.comm)
ksp_T.setOperators(A_T)
ksp_T.setType("preonly")
ksp_T.getPC().setType("lu")
ksp_T.getPC().setFactorSolverType("superlu_dist")
x_T = A_T.createVecRight()

# Set up files for visualisation
u_file = io.VTXWriter(msh.comm, "u.bp", [u_vis._cpp_object])
p_file = io.VTXWriter(msh.comm, "p.bp", [p_h._cpp_object])
ubar_file = io.VTXWriter(msh.comm, "ubar.bp", [ubar_n._cpp_object])
pbar_file = io.VTXWriter(msh.comm, "pbar.bp", [pbar_h._cpp_object])
T_file = io.VTXWriter(msh.comm, "T.bp", [T_n._cpp_object])
T_s_file = io.VTXWriter(msh.comm, "T_s.bp", [T_s_n._cpp_object])

t = 0.0
u_file.write(t)
p_file.write(t)
ubar_file.write(t)
pbar_file.write(t)
T_file.write(t)
T_s_file.write(t)
for n in range(num_time_steps):
    t += delta_t

    if solver_type == SolverType.NAVIER_STOKES:
        A.zeroEntries()
        fem.petsc.assemble_matrix_block(A, a, bcs=bcs)
        A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)

    u_h.x.array[:u_offset] = x.array_r[:u_offset]
    u_h.x.scatter_forward()
    p_h.x.array[:p_offset - u_offset] = x.array_r[u_offset:p_offset]
    p_h.x.scatter_forward()
    ubar_n.x.array[:ubar_offset -
                   p_offset] = x.array_r[p_offset:ubar_offset]
    ubar_n.x.scatter_forward()
    pbar_h.x.array[:(len(x.array_r) - ubar_offset)
                   ] = x.array_r[ubar_offset:]
    pbar_h.x.scatter_forward()
    # TODO
    # if len(neumann_bcs) == 0:
    #     p_h.x.array[:] -= domain_average(submesh_f, p_h)

    A_T.zeroEntries()
    fem.petsc.assemble_matrix_block(A_T, a_T)
    A_T.assemble()

    with b_T.localForm() as b_T_loc:
        b_T_loc.set(0)
    fem.petsc.assemble_vector_block(b_T, L_T, a_T)

    ksp_T.solve(b_T, x_T)
    offset_T = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    T_n.x.array[:offset_T] = x_T.array_r[:offset_T]
    T_n.x.scatter_forward()
    T_s_n.x.array[:(len(x_T.array_r) - offset_T)] = x_T.array_r[offset_T:]
    T_s_n.x.scatter_forward()

    u_vis.interpolate(u_n)

    u_file.write(t)
    p_file.write(t)
    ubar_file.write(t)
    pbar_file.write(t)
    T_file.write(t)
    T_s_file.write(t)

    # Update u_n
    u_n.x.array[:] = u_h.x.array

u_file.close()
p_file.close()
T_file.close()
T_s_file.close()

# Compute errors
e_div_u = norm_L2(msh.comm, div(u_h))
# This scheme conserves mass exactly, so check this
assert np.isclose(e_div_u, 0.0)
