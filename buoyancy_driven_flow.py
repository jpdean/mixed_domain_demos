# TODO This demo needs tidying and simplifying

# This demo solves a buoyancy driven flow problem. The domain is
# is a cuboid and contains a heated cylinder. The cylinder is
# surrounded by an incompressible fluid. The Navier--Stokes
# equations are solved in the fluid portion of the domain
# using the HDG scheme in hdg_navier_stokes.py. The thermal
# problem is solved over the entire domain. In the solid region,
# we use a standard conforming Galerkin method. In the fluid region,
# we use an upwind discontinuous Galerkin method. The schemes are
# coupled at the fluid-solid interface using Nitsche's method (see
# cg_dg_advec_diffusion.py).

import hdg_navier_stokes
from dolfinx import fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from ufl import (TrialFunction, TestFunction, CellDiameter, FacetNormal,
                 inner, grad, dx, avg, div, conditional,
                 gt, dot, Measure, as_vector)
from ufl import jump as jump_T
import gmsh
from utils import (convert_facet_tags, norm_L2, par_print,
                   compute_interface_integration_entities,
                   compute_interior_facet_integration_entities)


def generate_mesh(comm, h, cell_type=mesh.CellType.triangle):
    # Get geometric dimension of domain
    if cell_type == mesh.CellType.tetrahedron or \
            cell_type == mesh.CellType.hexahedron:
        d = 3
    else:
        d = 2

    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("model")
        factory = gmsh.model.geo

        # Set some parameters
        length = 0.8  # Domain length
        height = 1.25  # Domain height
        c = (0.41, 0.25)  # Centre of cylinder
        r = 0.05  # Radius of cylinder
        # Radius of square region surrounding cylinder for
        # meshing purposes
        r_s = 0.1

        # Corners of the domain
        rectangle_points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(length, 0.0, 0.0, h),
            factory.addPoint(length, height, 0.0, h),
            factory.addPoint(0.0, height, 0.0, h)
        ]

        # Create points to define the cylinder
        thetas = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4,
                  7 * np.pi / 4, 9 * np.pi / 4]
        circle_points = [factory.addPoint(c[0], c[1], 0.0)] + \
            [factory.addPoint(c[0] + r * np.cos(theta),
                              c[1] + r * np.sin(theta), 0.0)
                for theta in thetas]

        # Corners of a square surrounding the cylinder
        square_points = [
            factory.addPoint(c[0] + r_s * np.cos(theta),
                             c[1] + r_s * np.sin(theta), 0.0)
            for theta in thetas]

        # Some points to help define a refined region of the mesh
        # around the buoyant plume
        plume_points = [factory.addPoint(0.31, 1.0, 0.0, h),
                        factory.addPoint(0.51, 1.0, 0.0, h)]

        # Domain boundary
        rectangle_lines = [
            factory.addLine(rectangle_points[0], rectangle_points[1]),
            factory.addLine(rectangle_points[1], rectangle_points[2]),
            factory.addLine(rectangle_points[2], rectangle_points[3]),
            factory.addLine(rectangle_points[3], rectangle_points[0])
        ]

        # Cylinder boundary
        circle_lines = [
            factory.addCircleArc(
                circle_points[1], circle_points[0], circle_points[2]),
            factory.addCircleArc(
                circle_points[2], circle_points[0], circle_points[3]),
            factory.addCircleArc(
                circle_points[3], circle_points[0], circle_points[4]),
            factory.addCircleArc(
                circle_points[4], circle_points[0], circle_points[1])
        ]

        square_lines = [
            factory.addLine(square_points[0], square_points[1]),
            factory.addLine(square_points[1], square_points[2]),
            factory.addLine(square_points[2], square_points[3]),
            factory.addLine(square_points[3], square_points[0])]

        plume_lines = [square_lines[0],
                       factory.addLine(square_points[1], plume_points[0]),
                       factory.addLine(plume_points[0], plume_points[1]),
                       factory.addLine(plume_points[1], square_points[0])]

        # Define regions around the cylinder where the mesh is refined
        # to better capture the boundary layer
        bl_diag_lines = [
            factory.addLine(circle_points[i + 1], square_points[i])
            for i in range(4)]
        boundary_layer_lines = [
            [square_lines[0], - bl_diag_lines[1],
                - circle_lines[0], bl_diag_lines[0]],
            [square_lines[1], - bl_diag_lines[2],
                - circle_lines[1], bl_diag_lines[1]],
            [square_lines[2], - bl_diag_lines[3],
                - circle_lines[2], bl_diag_lines[2]],
            [square_lines[3], - bl_diag_lines[0],
                - circle_lines[3], bl_diag_lines[3]]
        ]

        # Create curves
        rectangle_curve = factory.addCurveLoop(rectangle_lines)
        circle_curve = factory.addCurveLoop(circle_lines)
        square_curve = factory.addCurveLoop(square_lines)
        boundary_layer_curves = [
            factory.addCurveLoop(bll) for bll in boundary_layer_lines]
        plume_curve = factory.add_curve_loop(plume_lines)

        # Create surfaces
        outer_surface = factory.addPlaneSurface(
            [rectangle_curve, square_curve, plume_curve])
        boundary_layer_surfaces = [
            factory.addPlaneSurface([blc])
            for blc in boundary_layer_curves]
        circle_surface = factory.addPlaneSurface([circle_curve])
        plume_surface = factory.addPlaneSurface([plume_curve])

        num_bl_eles_norm = round(0.3 * 1 / h)
        num_bl_eles_tan = round(0.8 * 1 / h)
        progression_coeff = 1.2
        for i in range(len(boundary_layer_surfaces)):
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][0], num_bl_eles_tan)
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][1], num_bl_eles_norm,
                coef=progression_coeff)
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][2], num_bl_eles_tan)
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][3], num_bl_eles_norm,
                coef=progression_coeff)
            gmsh.model.geo.mesh.setTransfiniteSurface(
                boundary_layer_surfaces[i])

        # The first plume line is already set, so only set others
        num_plume_eles = round(3.0 * 1 / h)
        gmsh.model.geo.mesh.setTransfiniteCurve(
            plume_lines[1], num_plume_eles)
        gmsh.model.geo.mesh.setTransfiniteCurve(
            plume_lines[2], num_bl_eles_tan)
        gmsh.model.geo.mesh.setTransfiniteCurve(
            plume_lines[3], num_plume_eles)
        gmsh.model.geo.mesh.setTransfiniteSurface(
            plume_surface)

        # Extrude the mesh in 3D
        if d == 3:
            if cell_type == mesh.CellType.tetrahedron:
                recombine = False
            else:
                recombine = True
            extrude_surfs = [(2, surf) for surf in [
                outer_surface] + boundary_layer_surfaces
                + [circle_surface] + [plume_surface]]
            gmsh.model.geo.extrude(
                extrude_surfs, 0, 0, 0.5, [4], recombine=recombine)

        factory.synchronize()

        # Define physical groups
        if d == 3:
            # FIXME Don't hardcode
            # FIXME Need to work these out again
            gmsh.model.addPhysicalGroup(
                3, [1, 2, 3, 4, 5, 7],
                volume_id["fluid"])
            gmsh.model.addPhysicalGroup(
                3, [6],
                volume_id["solid"])

            gmsh.model.addPhysicalGroup(
                2, [1, 2, 3, 4, 5, 7, 36, 40, 44, 48, 81,
                    213, 103, 125, 147, 169],
                boundary_id["walls"])
            # NOTE Does not include ends
            gmsh.model.addPhysicalGroup(
                2, [98, 120, 142, 164],
                boundary_id["obstacle"])
        else:
            gmsh.model.addPhysicalGroup(
                2, [outer_surface, plume_surface] + boundary_layer_surfaces,
                volume_id["fluid"])
            gmsh.model.addPhysicalGroup(
                2, [circle_surface], volume_id["solid"])

            gmsh.model.addPhysicalGroup(
                1, rectangle_lines,
                boundary_id["walls"])
            gmsh.model.addPhysicalGroup(
                1, circle_lines, boundary_id["obstacle"])

        gmsh.option.setNumber("Mesh.Smoothing", 25)
        if cell_type == mesh.CellType.quadrilateral \
                or cell_type == mesh.CellType.hexahedron:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)

        # gmsh.write("cyl_msh.msh")

        gmsh.model.mesh.generate(d)
        # gmsh.fltk.run()

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, ct, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=d, partitioner=partitioner)
    ft.name = "Facet markers"

    return msh, ct, ft, volume_id, boundary_id


def zero(x): return np.zeros_like(x[:msh.topology.dim])


# Simulation parameters
num_time_steps = 5  # 1280
t_end = 1  # 5
h = 0.04  # Maximum element diameter
k = 2  # Polynomial degree
solver_type = hdg_navier_stokes.SolverType.NAVIER_STOKES
delta_t_write = t_end / 100  # How often to write to file
g_y = -9.81  # Acceleration due to gravity

# Air
mu = 1.825e-5  # Dynamic viscosity
rho = 1.204  # Fluid density
eps = 3.43e-3  # Thermal expansion coefficient
f_T = 1e8  # Thermal source
kappa_f = 0.02514  # Thermal conductivity of fluid
kappa_s = 83.5  # Thermal conductivity of solid
rho_s = 7860  # Solid density
c_s = 462  # Solid specific heat
c_f = 1007  # Fluid specific heat

# Material parameters
# Water
# mu = 0.0010518  # Dynamic viscosity
# rho = 1000  # Fluid density
# g = as_vector((0.0, -9.81))
# eps = 0.000214  # Thermal expansion coefficient
# f_T = 10e6  # Thermal source
# kappa = 0.6  # Thermal conductivity
# rho_s = 7860  # Solid density
# c_s = 462  # Solid specific heat
# c_f = 4184  # Fluid specific heat

# Volume and boundary ids
volume_id = {"fluid": 1,
             "solid": 2}
boundary_id = {"walls": 2,
               "obstacle": 3}

# Create mesh
comm = MPI.COMM_WORLD
msh, ct, ft, volume_id, boundary_id = generate_mesh(
    comm, h=h, cell_type=mesh.CellType.quadrilateral)

# Create sub-meshes of fluid and solid domains
tdim = msh.topology.dim
submesh_f, sm_f_to_msh = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == volume_id["fluid"]])[:2]
submesh_s, sm_s_to_msh = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == volume_id["solid"]])[:2]

# Convert meshtags to fluid sub-mesh
fdim = tdim - 1
submesh_f.topology.create_connectivity(fdim, tdim)
ft_f = convert_facet_tags(msh, submesh_f, sm_f_to_msh, ft)

# Define boundary conditions for fluid solver
boundary_conditions = {"walls": (hdg_navier_stokes.BCType.Dirichlet, zero),
                       "obstacle": (hdg_navier_stokes.BCType.Dirichlet, zero)}

# Create function spaces for fluid problem
scheme = hdg_navier_stokes.Scheme.DRW
facet_mesh_f, fm_f_to_sm_f = hdg_navier_stokes.create_facet_mesh(submesh_f)
V_f, Q_f, Vbar_f, Qbar_f = hdg_navier_stokes.create_function_spaces(
    submesh_f, facet_mesh_f, scheme, k)

# Function spaces for fluid and solid temperature
Q = fem.FunctionSpace(submesh_f, ("Discontinuous Lagrange", k))
Q_s = fem.FunctionSpace(submesh_s, ("Lagrange", k))

# Cell and facet velocities at previous time step
u_n = fem.Function(V_f)
ubar_n = fem.Function(Vbar_f)
# Fluid and solid temperature at previous time step
T_n = fem.Function(Q)
T_s_n = fem.Function(Q_s)

# Buoyancy force (taking rho as reference density), see
# https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy)
# where I've omitted the rho g h part (can think of this is
# lumping gravity in with pressure, see 2P4) and taken
# T_0 to be 0
eps = fem.Constant(submesh_f, PETSc.ScalarType(eps))
# Acceleration due to gravity
if msh.topology.dim == 3:
    g = as_vector((0.0, g_y, 0.0))
else:
    g = as_vector((0.0, g_y))
# Buoyancy force
f = - eps * rho * T_n * g

# Time step
delta_t = t_end / num_time_steps  # TODO Make constant

# Create forms for fluid solver
nu = mu / rho  # Kinematic viscosity
a, L, bcs, bc_funcs = hdg_navier_stokes.create_forms(
    V_f, Q_f, Vbar_f, Qbar_f, submesh_f, k, delta_t, nu,
    fm_f_to_sm_f, solver_type, boundary_conditions,
    boundary_id, ft_f, f, facet_mesh_f, u_n, ubar_n)

# Trial and test function for fluid temperature
T, w = TrialFunction(Q), TestFunction(Q)
# Trial and test funcitons for the solid temperature
T_s, w_s = TrialFunction(Q_s), TestFunction(Q_s)

# Boundary conditions for the thermal solver
dirichlet_bcs_T = [(boundary_id["walls"], lambda x: np.zeros_like(x[0]))]

# Create entity maps for the thermal problem
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
msh_to_sm_f = np.full(num_cells, -1)
msh_to_sm_f[sm_f_to_msh] = np.arange(len(sm_f_to_msh))
msh_to_sm_s = np.full(num_cells, -1)
msh_to_sm_s[sm_s_to_msh] = np.arange(len(sm_s_to_msh))
entity_maps = {submesh_f: msh_to_sm_f,
               submesh_s: msh_to_sm_s}

# Create measure for integration. Assign the first (cell, local facet)
# pair to the cell in omega_0, corresponding to the "+" restriction. Assign
# the second pair to the omega_1 cell, corresponding to the "-" restriction.
fluid_int_facets = 7  # FIXME Don't hardcode
# facet_integration_entities = {boundary_id["obstacle"]: [],
#                               fluid_int_facets: []}
facet_imap = msh.topology.index_map(fdim)
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
domain_f_cells = ct.indices[ct.values == volume_id["fluid"]]
domain_s_cells = ct.indices[ct.values == volume_id["solid"]]
interface_facets = ft.indices[ft.values == boundary_id["obstacle"]]
obstacle_facet_entities, msh_to_sm_f, msh_to_sm_s = \
    compute_interface_integration_entities(
        interface_facets, domain_f_cells, domain_s_cells, c_to_f, f_to_c,
        facet_imap, msh_to_sm_f, msh_to_sm_s)

# FIXME Do this more efficiently
fluid_int_facet_entities = compute_interior_facet_integration_entities(
    submesh_f, sm_f_to_msh)
facet_integration_entities = [
    (boundary_id["obstacle"], obstacle_facet_entities),
    (fluid_int_facets, fluid_int_facet_entities)]

# Create measures for thermal problem
dx_T = Measure("dx", domain=msh, subdomain_data=ct)
ds_T = Measure("ds", domain=msh, subdomain_data=ft)
dS_T = Measure("dS", domain=msh,
               subdomain_data=facet_integration_entities)

h_T = CellDiameter(msh)
n_T = FacetNormal(msh)
lmbda_T = conditional(gt(dot(u_n, n_T), 0), 1, 0)
gamma_int = 32  # Penalty param for temperature on interface
alpha = 32.0 * k**2  # Penalty param for DG temp solver

# Fluid velocity at current time step
u_h = u_n.copy()

# Convert to Constants
delta_t = fem.Constant(msh, PETSc.ScalarType(delta_t))
alpha = fem.Constant(msh, PETSc.ScalarType(alpha))
gamma_int = fem.Constant(msh, PETSc.ScalarType(gamma_int))
kappa_f = fem.Constant(msh, PETSc.ScalarType(kappa_f))
kappa_s = fem.Constant(msh, PETSc.ScalarType(kappa_s))
rho_s = fem.Constant(submesh_s, PETSc.ScalarType(rho_s))
c_s = fem.Constant(submesh_s, PETSc.ScalarType(c_s))
c_f = fem.Constant(submesh_f, PETSc.ScalarType(c_f))

# Jump in kappa at interface dealt with using approach in DiPietro
# p. 150 sec 4.5
# Kappa harmonic mean
kappa_hm = 2 * kappa_f * kappa_s / (kappa_f + kappa_s)
# Weights for average operator
kappa_w_f = kappa_s / (kappa_f + kappa_s)
kappa_w_s = kappa_f / (kappa_f + kappa_s)

# Define forms for the thermal problem
a_T_00 = inner(rho * c_f * T / delta_t, w) * dx_T(volume_id["fluid"]) + \
    rho * c_f * (- inner(u_h * T, grad(w)) * dx_T(volume_id["fluid"]) +
                 inner(lmbda_T("+") * dot(u_h("+"), n_T("+")) * T("+") -
                       lmbda_T("-") * dot(u_h("-"), n_T("-")) * T("-"),
                       jump_T(w)) * dS_T(fluid_int_facets) +
                 inner(lmbda_T * dot(u_h, n_T) * T, w) * ds_T) + \
    kappa_f * (inner(grad(T), grad(w)) * dx_T(volume_id["fluid"]) -
               inner(avg(grad(T)), jump_T(w, n_T)) * dS_T(fluid_int_facets) -
               inner(jump_T(T, n_T), avg(grad(w))) * dS_T(fluid_int_facets) +
               (alpha / avg(h_T)) * inner(
        jump_T(T, n_T), jump_T(w, n_T)) * dS_T(fluid_int_facets)) \
    + kappa_hm * gamma_int / avg(h_T) * inner(
        T("+"), w("+")) * dS_T(boundary_id["obstacle"]) \
    + kappa_f * kappa_w_f * (
    - inner(dot(grad(T("+")), n_T("+")),
            w("+")) * dS_T(boundary_id["obstacle"])
    - inner(dot(grad(w("+")), n_T("+")),
            T("+")) * dS_T(boundary_id["obstacle"]))

a_T_01 = - kappa_hm * gamma_int / avg(h_T) * inner(
    T_s("-"), w("+")) * dS_T(boundary_id["obstacle"]) \
    + kappa_s * kappa_w_s * inner(dot(grad(T_s("-")), n_T("-")),
                                  w("+")) * dS_T(boundary_id["obstacle"]) \
    + kappa_f * kappa_w_f * inner(dot(grad(w("+")), n_T("+")),
                                  T_s("-")) * dS_T(boundary_id["obstacle"])

a_T_10 = - kappa_hm * gamma_int / avg(h_T) * inner(
    T("+"), w_s("-")) * dS_T(boundary_id["obstacle"]) \
    + kappa_f * kappa_w_f * inner(dot(grad(T("+")), n_T("+")),
                                  w_s("-")) * dS_T(boundary_id["obstacle"]) \
    + kappa_s * kappa_w_s * inner(dot(grad(w_s("-")), n_T("-")),
                                  T("+")) * dS_T(boundary_id["obstacle"])

a_T_11 = inner(rho_s * c_s * T_s / delta_t, w_s) * dx_T(volume_id["solid"]) \
    + kappa_s * inner(grad(T_s), grad(w_s)) * dx_T(volume_id["solid"]) \
    + kappa_hm * gamma_int / avg(h_T) * inner(
    T_s("-"), w_s("-")) * dS_T(boundary_id["obstacle"]) \
    + kappa_s * kappa_w_s * (
    - inner(dot(grad(T_s("-")), n_T("-")),
            w_s("-")) * dS_T(boundary_id["obstacle"])
    - inner(dot(grad(w_s("-")), n_T("-")),
            T_s("-")) * dS_T(boundary_id["obstacle"]))

L_T_0 = inner(rho * c_f * T_n / delta_t, w) * dx_T(volume_id["fluid"])

# Apply Dirichlet BCs for the thermal problem
for bc in dirichlet_bcs_T:
    T_D = fem.Function(Q)
    T_D.interpolate(bc[1])
    a_T_00 += kappa_f * (- inner(grad(T), w * n_T) * ds_T(bc[0]) -
                         inner(grad(w), T * n_T) * ds_T(bc[0]) +
                         (alpha / h_T) * inner(T, w) * ds_T(bc[0]))
    L_T_0 += - rho * c_f * inner((1 - lmbda_T) * dot(u_h, n_T) * T_D,
                                 w) * ds_T(bc[0]) + \
        kappa_f * (- inner(T_D * n_T, grad(w)) * ds_T(bc[0]) +
                   (alpha / h_T) * inner(T_D, w) * ds_T(bc[0]))

L_T_1 = inner(f_T, w_s) * dx_T(volume_id["solid"]) \
    + inner(rho_s * c_s * T_s_n / delta_t, w_s) * dx_T(volume_id["solid"])

# Compile forms
a_T_00 = fem.form(a_T_00, entity_maps=entity_maps)
a_T_01 = fem.form(a_T_01, entity_maps=entity_maps)
a_T_10 = fem.form(a_T_10, entity_maps=entity_maps)
a_T_11 = fem.form(a_T_11, entity_maps=entity_maps)

L_T_0 = fem.form(L_T_0, entity_maps=entity_maps)
L_T_1 = fem.form(L_T_1, entity_maps=entity_maps)

a_T = [[a_T_00, a_T_01],
       [a_T_10, a_T_11]]
L_T = [L_T_0, L_T_1]

# Assemble matrix and vector for thermal problem
A_T = fem.petsc.create_matrix_block(a_T)
b_T = fem.petsc.create_vector_block(L_T)

# Set-up matrix and vectors for fluid problem
if solver_type == hdg_navier_stokes.SolverType.NAVIER_STOKES:
    A = fem.petsc.create_matrix_block(a)
else:
    A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
    A.assemble()
b = fem.petsc.create_vector_block(L)
x = A.createVecRight()

# Set-up solver for thermal problem
ksp_T = PETSc.KSP().create(msh.comm)
ksp_T.setOperators(A_T)
ksp_T.setType("preonly")
ksp_T.getPC().setType("lu")
ksp_T.getPC().setFactorSolverType("superlu_dist")
x_T = A_T.createVecRight()

# Set-up solver for fluid problem
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()
opts["mat_mumps_icntl_6"] = 2
opts["mat_mumps_icntl_14"] = 100
ksp.setFromOptions()

# Set-up functions for visualisation
if scheme == hdg_navier_stokes.Scheme.RW:
    u_vis = fem.Function(V_f)
else:
    V_vis = fem.VectorFunctionSpace(
        submesh_f, ("Discontinuous Lagrange", k + 1))
    u_vis = fem.Function(V_vis)
u_vis.name = "u"
p_h = fem.Function(Q_f)
p_h.name = "p"
pbar_h = fem.Function(Qbar_f)
pbar_h.name = "pbar"

# Set up files for visualisation
vis_files = [io.VTXWriter(msh.comm, file_name, [func._cpp_object])
             for (file_name, func)
             in [("u.bp", u_vis), ("p.bp", p_h), ("ubar.bp", ubar_n),
                 ("pbar.bp", pbar_h), ("T.bp", T_n), ("T_s.bp", T_s_n)]]

t = 0.0
t_last_write = 0.0
for vis_file in vis_files:
    vis_file.write(t)
u_offset, p_offset, ubar_offset = hdg_navier_stokes.compute_offsets(
    V_f, Q_f, Vbar_f)
for n in range(num_time_steps):
    t += delta_t.value
    par_print(comm, f"t = {t}")

    # Assemble Navier--Stokes problem
    if solver_type == hdg_navier_stokes.SolverType.NAVIER_STOKES:
        A.zeroEntries()
        fem.petsc.assemble_matrix_block(A, a, bcs=bcs)
        A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

    # Compute Navier--Stokes solution
    ksp.solve(b, x)

    # Recover Navier--Stokes solution
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

    # Assemble thermal problem
    A_T.zeroEntries()
    fem.petsc.assemble_matrix_block(A_T, a_T)
    A_T.assemble()

    with b_T.localForm() as b_T_loc:
        b_T_loc.set(0)
    fem.petsc.assemble_vector_block(b_T, L_T, a_T)

    # Solver thermal problem
    ksp_T.solve(b_T, x_T)

    # Recover thermal solution
    offset_T = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    T_n.x.array[:offset_T] = x_T.array_r[:offset_T]
    T_n.x.scatter_forward()
    T_s_n.x.array[:(len(x_T.array_r) - offset_T)] = x_T.array_r[offset_T:]
    T_s_n.x.scatter_forward()

    # Interpolate for visualisation
    u_vis.interpolate(u_n)

    # Write solution to file
    if t - t_last_write > delta_t_write or \
            n == num_time_steps - 1:
        for vis_file in vis_files:
            vis_file.write(t)
        t_last_write = t

    # Update u_n
    u_n.x.array[:] = u_h.x.array

for vis_file in vis_files:
    vis_file.close()

# TODO Remove
par_print(comm, x.norm())

# Compute errors
e_div_u = norm_L2(msh.comm, div(u_h))
# This scheme conserves mass exactly, so check this
assert np.isclose(e_div_u, 0.0)
