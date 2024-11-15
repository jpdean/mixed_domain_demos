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
from ufl import (
    TrialFunction,
    TestFunction,
    CellDiameter,
    FacetNormal,
    inner,
    grad,
    avg,
    div,
    conditional,
    gt,
    dot,
    Measure,
    as_vector,
)
from ufl import jump as jump_T
import gmsh
from utils import (
    convert_facet_tags,
    norm_L2,
    par_print,
    interface_int_entities,
    compute_interior_facet_integration_entities,
)
from dolfinx.fem.petsc import (
    create_matrix_block,
    create_vector_block,
    assemble_matrix_block,
    assemble_vector_block,
)


def generate_mesh(comm, h, cell_type=mesh.CellType.triangle):
    # Get geometric dimension of domain
    if cell_type == mesh.CellType.tetrahedron or cell_type == mesh.CellType.hexahedron:
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
            factory.addPoint(0.0, height, 0.0, h),
        ]

        # Create points to define the cylinder
        thetas = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4, 9 * np.pi / 4]
        circle_points = [factory.addPoint(c[0], c[1], 0.0)] + [
            factory.addPoint(c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), 0.0)
            for theta in thetas
        ]

        # Corners of a square surrounding the cylinder
        square_points = [
            factory.addPoint(
                c[0] + r_s * np.cos(theta), c[1] + r_s * np.sin(theta), 0.0
            )
            for theta in thetas
        ]

        # Some points to help define a refined region of the mesh
        # around the buoyant plume
        plume_points = [
            factory.addPoint(0.31, 1.0, 0.0, h),
            factory.addPoint(0.51, 1.0, 0.0, h),
        ]

        # Domain boundary
        rectangle_lines = [
            factory.addLine(rectangle_points[0], rectangle_points[1]),
            factory.addLine(rectangle_points[1], rectangle_points[2]),
            factory.addLine(rectangle_points[2], rectangle_points[3]),
            factory.addLine(rectangle_points[3], rectangle_points[0]),
        ]

        # Cylinder boundary
        circle_lines = [
            factory.addCircleArc(circle_points[1], circle_points[0], circle_points[2]),
            factory.addCircleArc(circle_points[2], circle_points[0], circle_points[3]),
            factory.addCircleArc(circle_points[3], circle_points[0], circle_points[4]),
            factory.addCircleArc(circle_points[4], circle_points[0], circle_points[1]),
        ]

        square_lines = [
            factory.addLine(square_points[0], square_points[1]),
            factory.addLine(square_points[1], square_points[2]),
            factory.addLine(square_points[2], square_points[3]),
            factory.addLine(square_points[3], square_points[0]),
        ]

        plume_lines = [
            square_lines[0],
            factory.addLine(square_points[1], plume_points[0]),
            factory.addLine(plume_points[0], plume_points[1]),
            factory.addLine(plume_points[1], square_points[0]),
        ]

        # Define regions around the cylinder where the mesh is refined
        # to better capture the boundary layer
        bl_diag_lines = [
            factory.addLine(circle_points[i + 1], square_points[i]) for i in range(4)
        ]
        boundary_layer_lines = [
            [square_lines[0], -bl_diag_lines[1], -circle_lines[0], bl_diag_lines[0]],
            [square_lines[1], -bl_diag_lines[2], -circle_lines[1], bl_diag_lines[1]],
            [square_lines[2], -bl_diag_lines[3], -circle_lines[2], bl_diag_lines[2]],
            [square_lines[3], -bl_diag_lines[0], -circle_lines[3], bl_diag_lines[3]],
        ]

        # Create curves
        rectangle_curve = factory.addCurveLoop(rectangle_lines)
        circle_curve = factory.addCurveLoop(circle_lines)
        square_curve = factory.addCurveLoop(square_lines)
        boundary_layer_curves = [
            factory.addCurveLoop(bll) for bll in boundary_layer_lines
        ]
        plume_curve = factory.add_curve_loop(plume_lines)

        # Create surfaces
        outer_surface = factory.addPlaneSurface(
            [rectangle_curve, square_curve, plume_curve]
        )
        boundary_layer_surfaces = [
            factory.addPlaneSurface([blc]) for blc in boundary_layer_curves
        ]
        circle_surface = factory.addPlaneSurface([circle_curve])
        plume_surface = factory.addPlaneSurface([plume_curve])

        num_bl_eles_norm = round(0.3 * 1 / h)
        num_bl_eles_tan = round(0.8 * 1 / h)
        progression_coeff = 1.2
        for i in range(len(boundary_layer_surfaces)):
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][0], num_bl_eles_tan
            )
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][1], num_bl_eles_norm, coef=progression_coeff
            )
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][2], num_bl_eles_tan
            )
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][3], num_bl_eles_norm, coef=progression_coeff
            )
            gmsh.model.geo.mesh.setTransfiniteSurface(boundary_layer_surfaces[i])

        # The first plume line is already set, so only set others
        num_plume_eles = round(3.0 * 1 / h)
        gmsh.model.geo.mesh.setTransfiniteCurve(plume_lines[1], num_plume_eles)
        gmsh.model.geo.mesh.setTransfiniteCurve(plume_lines[2], num_bl_eles_tan)
        gmsh.model.geo.mesh.setTransfiniteCurve(plume_lines[3], num_plume_eles)
        gmsh.model.geo.mesh.setTransfiniteSurface(plume_surface)

        # Extrude the mesh in 3D
        if d == 3:
            if cell_type == mesh.CellType.tetrahedron:
                recombine = False
            else:
                recombine = True
            extrude_surfs = [
                (2, surf)
                for surf in [outer_surface]
                + boundary_layer_surfaces
                + [circle_surface]
                + [plume_surface]
            ]
            gmsh.model.geo.extrude(extrude_surfs, 0, 0, 0.5, [4], recombine=recombine)

        factory.synchronize()

        # Define physical groups
        if d == 3:
            # FIXME Don't hardcode
            # FIXME Need to work these out again
            gmsh.model.addPhysicalGroup(3, [1, 2, 3, 4, 5, 7], volume_id["fluid"])
            gmsh.model.addPhysicalGroup(3, [6], volume_id["solid"])

            gmsh.model.addPhysicalGroup(
                2,
                [1, 2, 3, 4, 5, 7, 36, 40, 44, 48, 81, 213, 103, 125, 147, 169],
                boundary_id["walls"],
            )
            # NOTE Does not include ends
            gmsh.model.addPhysicalGroup(2, [98, 120, 142, 164], boundary_id["obstacle"])
        else:
            gmsh.model.addPhysicalGroup(
                2,
                [outer_surface, plume_surface] + boundary_layer_surfaces,
                volume_id["fluid"],
            )
            gmsh.model.addPhysicalGroup(2, [circle_surface], volume_id["solid"])

            gmsh.model.addPhysicalGroup(1, rectangle_lines, boundary_id["walls"])
            gmsh.model.addPhysicalGroup(1, circle_lines, boundary_id["obstacle"])

        gmsh.option.setNumber("Mesh.Smoothing", 25)
        if (
            cell_type == mesh.CellType.quadrilateral
            or cell_type == mesh.CellType.hexahedron
        ):
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)

        # gmsh.write("cyl_msh.msh")

        gmsh.model.mesh.generate(d)
        # gmsh.fltk.run()

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, ct, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=d, partitioner=partitioner
    )
    ft.name = "Facet markers"

    return msh, ct, ft, volume_id, boundary_id


def zero(x):
    return np.zeros_like(x[: msh.topology.dim])


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
volume_id = {"fluid": 1, "solid": 2}
boundary_id = {"walls": 1, "obstacle": 2}

# Define boundary conditions for fluid solver
boundary_conditions = {
    "walls": (hdg_navier_stokes.BCType.Dirichlet, zero),
    "obstacle": (hdg_navier_stokes.BCType.Dirichlet, zero),
}

# Boundary conditions for the thermal solver
dirichlet_bcs_T = {"walls": lambda x: np.zeros_like(x[0])}

# Create mesh
comm = MPI.COMM_WORLD
msh, ct, ft, volume_id, boundary_id = generate_mesh(
    comm, h=h, cell_type=mesh.CellType.quadrilateral
)

# Create sub-meshes of fluid and solid domains
tdim = msh.topology.dim
submesh_f, sm_f_to_msh = mesh.create_submesh(msh, tdim, ct.find(volume_id["fluid"]))[:2]
submesh_s, sm_s_to_msh = mesh.create_submesh(msh, tdim, ct.find(volume_id["solid"]))[:2]

# Create function spaces for Navier-Stokes problem
scheme = hdg_navier_stokes.Scheme.DRW
facet_mesh_f, fm_f_to_sm_f = hdg_navier_stokes.create_facet_mesh(submesh_f)
V, Q, Vbar, Qbar = hdg_navier_stokes.create_function_spaces(
    submesh_f, facet_mesh_f, scheme, k
)

# Function spaces for fluid and solid temperature
W_f = fem.functionspace(submesh_f, ("Discontinuous Lagrange", k))
W_s = fem.functionspace(submesh_s, ("Lagrange", k))

# Cell and facet velocities at previous time step
u_n = fem.Function(V)
ubar_n = fem.Function(Vbar)
# Fluid and solid temperature at previous time step
T_f_n = fem.Function(W_f)
T_s_n = fem.Function(W_s)

# Time step
delta_t = t_end / num_time_steps  # TODO Make constant


# Create forms for Navier-Stokes solver. We begin by defining the
# buoyancy force (taking rho as reference density), see
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
f = -eps * rho * T_f_n * g
nu = mu / rho  # Kinematic viscosity
fdim = tdim - 1
submesh_f.topology.create_connectivity(fdim, tdim)
ft_f = convert_facet_tags(msh, submesh_f, sm_f_to_msh, ft)
a, L, bcs, bc_funcs = hdg_navier_stokes.create_forms(
    V,
    Q,
    Vbar,
    Qbar,
    submesh_f,
    k,
    delta_t,
    nu,
    fm_f_to_sm_f,
    solver_type,
    boundary_conditions,
    boundary_id,
    ft_f,
    f,
    facet_mesh_f,
    u_n,
    ubar_n,
)

# Trial and test function for fluid temperature
T_f, w_f = TrialFunction(W_f), TestFunction(W_f)
# Trial and test functions for the solid temperature
T_s, w_s = TrialFunction(W_s), TestFunction(W_s)

# Create entity maps for the thermal problem. Since we take msh to be the
# integration domain, we must create maps from cells in msh to cells in
# submesh_f
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
msh_to_sm_f = np.full(num_cells, -1)
msh_to_sm_f[sm_f_to_msh] = np.arange(len(sm_f_to_msh))
msh_to_sm_s = np.full(num_cells, -1)
msh_to_sm_s[sm_s_to_msh] = np.arange(len(sm_s_to_msh))
entity_maps = {submesh_f: msh_to_sm_f, submesh_s: msh_to_sm_s}

# Create integration entities for the interface integral
interface_facets = ft.find(boundary_id["obstacle"])
domain_f_cells = ct.find(volume_id["fluid"])
domain_s_cells = ct.find(volume_id["solid"])
(
    obstacle_facet_entities,
    msh_to_sm_f,
    msh_to_sm_s,
) = interface_int_entities(
    msh, interface_facets, domain_f_cells, domain_s_cells, msh_to_sm_f, msh_to_sm_s
)

# Create integration entities for the interior facet integral
fluid_int_facet_entities = compute_interior_facet_integration_entities(
    submesh_f, sm_f_to_msh
)
fluid_int_facets = 3
facet_integration_entities = [
    (boundary_id["obstacle"], obstacle_facet_entities),
    (fluid_int_facets, fluid_int_facet_entities),
]

# Create measures for thermal problem
dx_T = Measure("dx", domain=msh, subdomain_data=ct)
ds_T = Measure("ds", domain=msh, subdomain_data=ft)
dS_T = Measure("dS", domain=msh, subdomain_data=facet_integration_entities)

# Define some quantities used in the finite element forms
h_T = CellDiameter(msh)
n_T = FacetNormal(msh)
# Marker for outflow boundaries
lmbda_T = conditional(gt(dot(u_n, n_T), 0), 1, 0)
gamma_int = 32  # Penalty param for temperature on interface
alpha = 32.0 * k**2  # Penalty param for DG temp solver
u_h = u_n.copy()  # Fluid velocity at current time step

# Convert to Constants
delta_t = fem.Constant(msh, PETSc.ScalarType(delta_t))
alpha = fem.Constant(msh, PETSc.ScalarType(alpha))
gamma_int = fem.Constant(msh, PETSc.ScalarType(gamma_int))
kappa_f = fem.Constant(msh, PETSc.ScalarType(kappa_f))
kappa_s = fem.Constant(msh, PETSc.ScalarType(kappa_s))
rho_s = fem.Constant(submesh_s, PETSc.ScalarType(rho_s))
c_s = fem.Constant(submesh_s, PETSc.ScalarType(c_s))
c_f = fem.Constant(submesh_f, PETSc.ScalarType(c_f))

# Define some quantities that are used to handle the discontinuity in
# kappa at the interface (see DiPietro Sec 4.5 p. 150)
# Kappa harmonic mean
kappa_hm = 2 * kappa_f * kappa_s / (kappa_f + kappa_s)
# Weights for weighted average operator
kappa_w_f = kappa_s / (kappa_f + kappa_s)
kappa_w_s = kappa_f / (kappa_f + kappa_s)

# Define forms for the thermal problem
# FIXME Refactor cg_dg_advec_diffusion.py and use forms in this code to avoid
# duplication
a_T_00 = (
    inner(rho * c_f * T_f / delta_t, w_f) * dx_T(volume_id["fluid"])
    + rho
    * c_f
    * (
        -inner(u_h * T_f, grad(w_f)) * dx_T(volume_id["fluid"])
        + inner(
            lmbda_T("+") * dot(u_h("+"), n_T("+")) * T_f("+")
            - lmbda_T("-") * dot(u_h("-"), n_T("-")) * T_f("-"),
            jump_T(w_f),
        )
        * dS_T(fluid_int_facets)
        + inner(lmbda_T * dot(u_h, n_T) * T_f, w_f) * ds_T
    )
    + kappa_f
    * (
        inner(grad(T_f), grad(w_f)) * dx_T(volume_id["fluid"])
        - inner(avg(grad(T_f)), jump_T(w_f, n_T)) * dS_T(fluid_int_facets)
        - inner(jump_T(T_f, n_T), avg(grad(w_f))) * dS_T(fluid_int_facets)
        + (alpha / avg(h_T))
        * inner(jump_T(T_f, n_T), jump_T(w_f, n_T))
        * dS_T(fluid_int_facets)
    )
    + kappa_hm
    * gamma_int
    / avg(h_T)
    * inner(T_f("+"), w_f("+"))
    * dS_T(boundary_id["obstacle"])
    + kappa_f
    * kappa_w_f
    * (
        -inner(dot(grad(T_f("+")), n_T("+")), w_f("+")) * dS_T(boundary_id["obstacle"])
        - inner(dot(grad(w_f("+")), n_T("+")), T_f("+")) * dS_T(boundary_id["obstacle"])
    )
)

a_T_01 = (
    -kappa_hm
    * gamma_int
    / avg(h_T)
    * inner(T_s("-"), w_f("+"))
    * dS_T(boundary_id["obstacle"])
    + kappa_s
    * kappa_w_s
    * inner(dot(grad(T_s("-")), n_T("-")), w_f("+"))
    * dS_T(boundary_id["obstacle"])
    + kappa_f
    * kappa_w_f
    * inner(dot(grad(w_f("+")), n_T("+")), T_s("-"))
    * dS_T(boundary_id["obstacle"])
)

a_T_10 = (
    -kappa_hm
    * gamma_int
    / avg(h_T)
    * inner(T_f("+"), w_s("-"))
    * dS_T(boundary_id["obstacle"])
    + kappa_f
    * kappa_w_f
    * inner(dot(grad(T_f("+")), n_T("+")), w_s("-"))
    * dS_T(boundary_id["obstacle"])
    + kappa_s
    * kappa_w_s
    * inner(dot(grad(w_s("-")), n_T("-")), T_f("+"))
    * dS_T(boundary_id["obstacle"])
)

a_T_11 = (
    inner(rho_s * c_s * T_s / delta_t, w_s) * dx_T(volume_id["solid"])
    + kappa_s * inner(grad(T_s), grad(w_s)) * dx_T(volume_id["solid"])
    + kappa_hm
    * gamma_int
    / avg(h_T)
    * inner(T_s("-"), w_s("-"))
    * dS_T(boundary_id["obstacle"])
    + kappa_s
    * kappa_w_s
    * (
        -inner(dot(grad(T_s("-")), n_T("-")), w_s("-")) * dS_T(boundary_id["obstacle"])
        - inner(dot(grad(w_s("-")), n_T("-")), T_s("-")) * dS_T(boundary_id["obstacle"])
    )
)

L_T_0 = inner(rho * c_f * T_f_n / delta_t, w_f) * dx_T(volume_id["fluid"])

# Apply Dirichlet BCs for the thermal problem
for b_name, bc_func in dirichlet_bcs_T.items():
    b_id = boundary_id[b_name]
    T_D = fem.Function(W_f)
    T_D.interpolate(bc_func)
    a_T_00 += kappa_f * (
        -inner(grad(T_f), w_f * n_T) * ds_T(b_id)
        - inner(grad(w_f), T_f * n_T) * ds_T(b_id)
        + (alpha / h_T) * inner(T_f, w_f) * ds_T(b_id)
    )
    L_T_0 += -rho * c_f * inner((1 - lmbda_T) * dot(u_h, n_T) * T_D, w_f) * ds_T(
        b_id
    ) + kappa_f * (
        -inner(T_D * n_T, grad(w_f)) * ds_T(b_id)
        + (alpha / h_T) * inner(T_D, w_f) * ds_T(b_id)
    )

L_T_1 = inner(f_T, w_s) * dx_T(volume_id["solid"]) + inner(
    rho_s * c_s * T_s_n / delta_t, w_s
) * dx_T(volume_id["solid"])

# Compile forms for the thermal problem
a_T_00 = fem.form(a_T_00, entity_maps=entity_maps)
a_T_01 = fem.form(a_T_01, entity_maps=entity_maps)
a_T_10 = fem.form(a_T_10, entity_maps=entity_maps)
a_T_11 = fem.form(a_T_11, entity_maps=entity_maps)
L_T_0 = fem.form(L_T_0, entity_maps=entity_maps)
L_T_1 = fem.form(L_T_1, entity_maps=entity_maps)

# Define block structure for thermal problem
a_T = [[a_T_00, a_T_01], [a_T_10, a_T_11]]
L_T = [L_T_0, L_T_1]

# Assemble matrix and vector for thermal problem
A_T = create_matrix_block(a_T)
b_T = create_vector_block(L_T)

# Set-up matrix and vectors for fluid problem
if solver_type == hdg_navier_stokes.SolverType.NAVIER_STOKES:
    A = create_matrix_block(a)
else:
    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()
b = create_vector_block(L)
x = A.createVecRight()

# Set-up solver for thermal problem
ksp_T = PETSc.KSP().create(msh.comm)
ksp_T.setOperators(A_T)
ksp_T.setType("preonly")
ksp_T.getPC().setType("lu")
ksp_T.getPC().setFactorSolverType("superlu_dist")
x_T = A_T.createVecRight()

# Set-up solver for Navier-Stokes problem
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
    u_vis = fem.Function(V)
else:
    V_vis = fem.functionspace(
        submesh_f, ("Discontinuous Lagrange", k + 1, (msh.geometry.dim,))
    )
    u_vis = fem.Function(V_vis)
u_vis.name = "u"
p_h = fem.Function(Q)
p_h.name = "p"
pbar_h = fem.Function(Qbar)
pbar_h.name = "pbar"

# Set-up files for visualisation
vis_files = [
    io.VTXWriter(msh.comm, file_name, [func._cpp_object], "BP4")
    for (file_name, func) in [
        ("u.bp", u_vis),
        ("p.bp", p_h),
        ("ubar.bp", ubar_n),
        ("pbar.bp", pbar_h),
        ("T.bp", T_f_n),
        ("T_s.bp", T_s_n),
    ]
]

# Time-stepping loop
t = 0.0
t_last_write = 0.0
for vis_file in vis_files:
    vis_file.write(t)
u_offset, p_offset, ubar_offset = hdg_navier_stokes.compute_offsets(V, Q, Vbar)
for n in range(num_time_steps):
    t += delta_t.value
    par_print(comm, f"t = {t}")

    # Assemble Navier-Stokes problem
    if solver_type == hdg_navier_stokes.SolverType.NAVIER_STOKES:
        A.zeroEntries()
        assemble_matrix_block(A, a, bcs=bcs)
        A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    assemble_vector_block(b, L, a, bcs=bcs)

    # Compute Navier-Stokes solution
    ksp.solve(b, x)

    # Recover Navier-Stokes solution
    u_h.x.array[:u_offset] = x.array_r[:u_offset]
    u_h.x.scatter_forward()
    p_h.x.array[: p_offset - u_offset] = x.array_r[u_offset:p_offset]
    p_h.x.scatter_forward()
    ubar_n.x.array[: ubar_offset - p_offset] = x.array_r[p_offset:ubar_offset]
    ubar_n.x.scatter_forward()
    pbar_h.x.array[: (len(x.array_r) - ubar_offset)] = x.array_r[ubar_offset:]
    pbar_h.x.scatter_forward()

    # Assemble thermal problem
    A_T.zeroEntries()
    assemble_matrix_block(A_T, a_T)
    A_T.assemble()

    with b_T.localForm() as b_T_loc:
        b_T_loc.set(0)
    assemble_vector_block(b_T, L_T, a_T)

    # Solver thermal problem
    ksp_T.solve(b_T, x_T)

    # Recover thermal solution
    offset_T = W_f.dofmap.index_map.size_local * W_f.dofmap.index_map_bs
    T_f_n.x.array[:offset_T] = x_T.array_r[:offset_T]
    T_f_n.x.scatter_forward()
    T_s_n.x.array[: (len(x_T.array_r) - offset_T)] = x_T.array_r[offset_T:]
    T_s_n.x.scatter_forward()

    # Interpolate for visualisation
    u_vis.interpolate(u_n)

    # Write solution to file
    if t - t_last_write > delta_t_write or n == num_time_steps - 1:
        for vis_file in vis_files:
            vis_file.write(t)
        t_last_write = t

    # Update u_n
    u_n.x.array[:] = u_h.x.array

for vis_file in vis_files:
    vis_file.close()

# Compute errors
e_div_u = norm_L2(msh.comm, div(u_h))
# This scheme conserves mass exactly, so check this
assert np.isclose(e_div_u, 0.0)
