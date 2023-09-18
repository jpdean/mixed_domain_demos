# This demo solves the equations of magnetohydrodynamics for a incompressible
# liquid metal flowing through a square channel with an applied transverse
# magnetic field. The domain contains two regions: the fluid region and the
# solid region consisting of the Hartmann walls (of finite
# thickness and conductivity). The sidewalls are assumed to be perfectly
# insulating. We use the HDG scheme given in "Hybridized discontinuous Galerkin
# methods for incompressible flows on meshes with quadrilateral cells" by
# J. P. Dean, S. Rhebergen, and G. N. Well. to solve
# the incompressible Navier-Stokes equations. We solve Maxwell's equations
# using a scheme similar to "A fully divergence-free finite element method
# for magnetohydrodynamic equations" by Hiptmair et al. See also Sec. 3.6.3
# in https://academic.oup.com/book/5953/chapter/149296535?login=true. Our
# approach is fully coupled and yields exactly divergence free velocity field
# and magnetic induction.

import hdg_navier_stokes
from hdg_navier_stokes import SolverType, Scheme, TimeDependentExpression
from mpi4py import MPI
import gmsh
import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.io import gmshio
from hdg_navier_stokes import BCType
from petsc4py import PETSc
from utils import (norm_L2, normal_jump_error, convert_facet_tags, par_print,
                   compute_cell_boundary_integration_entities)
import ufl
from ufl import (div, TrialFunction, TestFunction, inner, curl, cross,
                 as_vector, grad, outer, dot)
from dolfinx.cpp.fem import compute_integration_domains


def solve(solver_type, k, nu, num_time_steps, delta_t, scheme, msh, ct, ft,
          volumes, boundaries, boundary_conditions, f, u_i_expr, sigma_s,
          sigma_f, mu):
    # Create a sub-mesh of the fluid region
    submesh_f, sm_f_to_msh = mesh.create_submesh(
        msh, msh.topology.dim, ct.find(volumes["fluid"]))[:2]

    # Create a facet sub-mesh of the fluid sub-mesh for the HDG Navier-Stokes
    # solver
    facet_mesh_f, fm_f_to_sm_f = hdg_navier_stokes.create_facet_mesh(submesh_f)

    # Create function spaces for Navier-Stokes solver
    V, Q, Vbar, Qbar = hdg_navier_stokes.create_function_spaces(
        submesh_f, facet_mesh_f, scheme, k)

    # H(curl; Ω)-conforming space for magnetic vector potential and electric
    # field
    X = fem.FunctionSpace(msh, ("Nedelec 1st kind H(curl)", k + 1))

    # Create function space for coefficients
    V_coeff = fem.FunctionSpace(msh, ("Discontinuous Lagrange", 0))

    # Create integration entities
    cell_boundaries = 0
    cell_boundary_facets = compute_cell_boundary_integration_entities(
        submesh_f)
    ft_f = convert_facet_tags(msh, submesh_f, sm_f_to_msh, ft)
    facet_integration_entities = [(cell_boundaries, cell_boundary_facets)]
    facet_integration_entities += compute_integration_domains(
        fem.IntegralType.exterior_facet, ft_f._cpp_object)

    # Define integration measures
    dx = ufl.Measure("dx", domain=msh)
    dx_c = ufl.Measure("dx", domain=submesh_f)
    # FIXME Figure out why this is being estimated wrong for DRW
    # NOTE k**2 works on affine meshes
    quad_deg = (k + 1)**2
    ds_c = ufl.Measure(
        "ds", subdomain_data=facet_integration_entities, domain=submesh_f,
        metadata={"quadrature_degree": quad_deg})
    dx_f = ufl.Measure("dx", domain=facet_mesh_f)

    # Create entity maps
    tdim = msh.topology.dim
    fdim = tdim - 1
    facet_imap_sm_f = msh.topology.index_map(fdim)
    num_facets_sm_f = facet_imap_sm_f.size_local + facet_imap_sm_f.num_ghosts
    sm_f_to_fm_f = np.full(num_facets_sm_f, -1)
    sm_f_to_fm_f[fm_f_to_sm_f] = np.arange(len(fm_f_to_sm_f))
    entity_maps = {facet_mesh_f: sm_f_to_fm_f,
                   msh: sm_f_to_msh}

    # Trial and test functions for the magnetic vector potential
    A, phi = TrialFunction(X), TestFunction(X)
    A_h = fem.Function(X)
    A_n = fem.Function(X)

    # Trial and test functions
    u, v = ufl.TrialFunction(V),  ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    ubar, vbar = ufl.TrialFunction(Vbar), ufl.TestFunction(Vbar)
    pbar, qbar = ufl.TrialFunction(Qbar), ufl.TestFunction(Qbar)

    # Define forms
    # Cell and facet velocities at previous time step
    u_n = fem.Function(V)
    ubar_n = fem.Function(Vbar)
    # Interpolate initial condition
    u_n.interpolate(u_i_expr)
    ubar_n.interpolate(u_i_expr)
    h = ufl.CellDiameter(submesh_f)  # TODO Fix for high order geom!
    n = ufl.FacetNormal(submesh_f)
    gamma = 64.0 * k**2 / h  # Scaled penalty param
    # Marker for outflow boundaries
    lmbda = ufl.conditional(ufl.lt(dot(u_n, n), 0), 1, 0)
    delta_t = fem.Constant(msh, PETSc.ScalarType(delta_t))
    nu = fem.Constant(submesh_f, PETSc.ScalarType(nu))
    # Conductivity
    sigma = fem.Function(V_coeff)
    sigma.interpolate(
        lambda x: np.full_like(x[0], sigma_f), ct.find(volumes["fluid"]))
    sigma.interpolate(
        lambda x: np.full_like(x[0], sigma_s), ct.find(volumes["solid"]))
    # Permeability
    mu = fem.Constant(msh, mu)
    # Externally applied magnetic induction
    B_0 = as_vector((0, 1, 0))

    # Left-hand side diffusive terms
    a_00 = inner(u / delta_t, v) * dx_c \
        + nu * inner(grad(u), grad(v)) * dx_c \
        - nu * inner(grad(u), outer(v, n)) * ds_c(cell_boundaries) \
        + nu * gamma * inner(outer(u, n),
                             outer(v, n)) * ds_c(cell_boundaries) \
        - nu * inner(outer(u, n), grad(v)) * ds_c(cell_boundaries)
    a_01 = fem.form(- inner(p * ufl.Identity(msh.topology.dim),
                    grad(v)) * dx_c)
    a_02 = - nu * gamma * inner(
        outer(ubar, n), outer(v, n)) * ds_c(cell_boundaries) \
        + nu * inner(outer(ubar, n), grad(v)) * ds_c(cell_boundaries)
    a_03 = fem.form(inner(pbar * ufl.Identity(msh.topology.dim),
                          outer(v, n)) * ds_c(cell_boundaries),
                    entity_maps=entity_maps)
    a_10 = fem.form(inner(u, grad(q)) * dx_c -
                    inner(dot(u, n), q) * ds_c(cell_boundaries))
    a_20 = - nu * inner(grad(u), outer(vbar, n)) * ds_c(cell_boundaries) \
        + nu * gamma * inner(outer(u, n), outer(vbar, n)
                             ) * ds_c(cell_boundaries)
    a_30 = fem.form(inner(dot(u, n), qbar) *
                    ds_c(cell_boundaries), entity_maps=entity_maps)
    a_23 = fem.form(
        inner(pbar * ufl.Identity(tdim), outer(vbar, n)) *
        ds_c(cell_boundaries),
        entity_maps=entity_maps)
    # On the Dirichlet boundary, the contribution from this term will be
    # added to the RHS in apply_lifting
    a_32 = fem.form(- inner(dot(ubar, n), qbar) * ds_c,
                    entity_maps=entity_maps)
    a_22 = - nu * gamma * \
        inner(outer(ubar, n), outer(vbar, n)) * ds_c(cell_boundaries)

    # LHS advective terms
    if solver_type == SolverType.NAVIER_STOKES:
        a_00 += - inner(outer(u, u_n), grad(v)) * dx_c \
            + inner(outer(u, u_n), outer(v, n)) * ds_c(cell_boundaries) \
            - inner(outer(u, lmbda * u_n), outer(v, n)) * ds_c(cell_boundaries)
        a_02 += inner(outer(ubar, lmbda * u_n), outer(v, n)) * \
            ds_c(cell_boundaries)
        a_20 += inner(outer(u, u_n), outer(vbar, n)) * ds_c(cell_boundaries) \
            - inner(outer(u, lmbda * u_n), outer(vbar, n)) * \
            ds_c(cell_boundaries)
        a_22 += inner(outer(ubar, lmbda * u_n),
                      outer(vbar, n)) * ds_c(cell_boundaries)

    # Add LHS terms from Maxwell's equations
    a_44 = fem.form(inner(sigma * A / delta_t, phi) * dx
                    + inner(1 / mu * curl(A), curl(phi)) * dx)
    a_40 = fem.form(inner(sigma * cross(B_0, u), phi) * dx_c
                    + inner(sigma * cross(curl(A_n), u), phi) * dx_c,
                    entity_maps={msh: sm_f_to_msh})
    a_04 = fem.form(
        inner(sigma * A / delta_t, cross(curl(A_n), v)) * dx_c
        - inner(sigma * cross(u_n, curl(A)), cross(curl(A_n), v)) * dx_c
        + inner(sigma * A / delta_t, cross(B_0, v)) * dx_c,
        entity_maps={msh: sm_f_to_msh})

    # Right-hand side terms
    L_0 = inner(f + u_n / delta_t, v) * dx_c \
        + inner(sigma * A_n / delta_t, cross(curl(A_n), v)) * dx_c \
        + inner(sigma * A_n / delta_t, cross(B_0, v)) * dx_c \
        + inner(sigma * cross(u_n, B_0), cross(B_0, v)) * dx_c
    L_1 = inner(fem.Constant(msh, 0.0), q) * dx_c
    L_2 = inner(fem.Constant(submesh_f, [PETSc.ScalarType(0.0)
                                         for i in range(tdim)]),
                vbar) * ds_c(cell_boundaries)
    L_3 = inner(fem.Constant(facet_mesh_f, PETSc.ScalarType(0.0)), qbar) * dx_f
    L_4 = inner(sigma * A_n / delta_t, phi) * dx

    # Boundary conditions
    # Fluid problem
    bcs = []
    bc_funcs = []  # FIXME Can now simplify this by getting func through bc
    for name, bc in boundary_conditions["u"].items():
        id = boundaries[name]
        bc_type, bc_expr = bc
        bc_func = fem.Function(Vbar)
        bc_func.interpolate(bc_expr)
        bc_funcs.append((bc_func, bc_expr))
        if bc_type == BCType.Dirichlet:
            facets = sm_f_to_fm_f[ft_f.indices[ft_f.values == id]]
            dofs = fem.locate_dofs_topological(Vbar, fdim, facets)
            bcs.append(fem.dirichletbc(bc_func, dofs))
        else:
            assert bc_type == BCType.Neumann
            L_2 += inner(bc_func, vbar) * ds_c(id)
            if solver_type == SolverType.NAVIER_STOKES:
                a_22 += - inner((1 - lmbda) * dot(ubar_n, n) *
                                ubar, vbar) * ds_c(id)
    # Electromagnetic problem
    for name, bc in boundary_conditions["A"].items():
        id = boundaries[name]
        bc_type, bc_expr = bc
        assert bc_type == BCType.Dirichlet
        bc_func = fem.Function(X)
        bc_func.interpolate(bc_expr)
        facets = ft.find(id)
        dofs = fem.locate_dofs_topological(X, fdim, facets)
        bcs.append(fem.dirichletbc(bc_func, dofs))

    # Compile forms
    a_00 = fem.form(a_00)
    a_02 = fem.form(a_02, entity_maps=entity_maps)
    a_20 = fem.form(a_20, entity_maps=entity_maps)
    a_22 = fem.form(a_22, entity_maps=entity_maps)

    L_0 = fem.form(L_0, entity_maps={msh: sm_f_to_msh})
    L_1 = fem.form(L_1)
    L_2 = fem.form(L_2, entity_maps=entity_maps)
    L_3 = fem.form(L_3)
    L_4 = fem.form(L_4)

    # Define block structure
    a = [[a_00, a_01, a_02, a_03, a_04],
         [a_10, None, None, None, None],
         [a_20, None, a_22, a_23, None],
         [a_30, None, a_32, None, None],
         [a_40, None, None, None, a_44]]
    L = [L_0, L_1, L_2, L_3, L_4]

    # Set-up matrix and vectors
    if solver_type == SolverType.NAVIER_STOKES:
        A = fem.petsc.create_matrix_block(a)
    else:
        A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
        A.assemble()
    b = fem.petsc.create_vector_block(L)
    x = A.createVecRight()

    # Set-up functions for visualisation (fluid problem)
    if scheme == Scheme.RW:
        u_vis = fem.Function(V)
    else:
        V_vis = fem.VectorFunctionSpace(
            submesh_f, ("Discontinuous Lagrange", k + 1))
        u_vis = fem.Function(V_vis)
    u_vis.name = "u"
    u_vis.interpolate(u_n)
    p_h = fem.Function(Q)
    p_h.name = "p"
    pbar_h = fem.Function(Qbar)
    pbar_h.name = "pbar"

    # Set-up functions for visualisation (Maxwell problem)
    X_vis = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k + 1))
    B_h = B_0 + curl(A_h)
    # B_h = curl(A_h)
    B_expr = fem.Expression(B_h, X_vis.element.interpolation_points())
    B_vis = fem.Function(X_vis)
    B_vis.interpolate(B_expr)

    # Configure solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    opts["mat_mumps_icntl_6"] = 2
    opts["mat_mumps_icntl_14"] = 100
    ksp.setFromOptions()

    # Set up files for visualisation
    vis_files = [io.VTXWriter(msh.comm, file_name, [func._cpp_object])
                 for (file_name, func)
                 in [("u.bp", u_vis), ("p.bp", p_h), ("ubar.bp", ubar_n),
                 ("pbar.bp", pbar_h), ("B.bp", B_vis)]]

    # Time-stepping loop
    t = 0.0
    for vis_file in vis_files:
        vis_file.write(t)
    u_offset, p_offset, ubar_offset = hdg_navier_stokes.compute_offsets(
        V, Q, Vbar)
    pbar_offset = ubar_offset + Qbar.dofmap.index_map.size_local * \
        Qbar.dofmap.index_map_bs
    for n in range(num_time_steps):
        t += delta_t.value
        par_print(comm, f"t = {t}")

        # Update time dependent expressions
        for bc_func, bc_expr in bc_funcs:
            if isinstance(bc_expr, TimeDependentExpression):
                bc_expr.t = t
                bc_func.interpolate(bc_expr)

        # Assemble matrix
        if solver_type == SolverType.NAVIER_STOKES:
            A.zeroEntries()
            fem.petsc.assemble_matrix_block(A, a, bcs=bcs)
            A.assemble()

        # Assemble vector
        with b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

        # Compute solution
        ksp.solve(b, x)

        # Recover solution
        u_n.x.array[:u_offset] = x.array_r[:u_offset]
        u_n.x.scatter_forward()
        p_h.x.array[:p_offset - u_offset] = x.array_r[u_offset:p_offset]
        p_h.x.scatter_forward()
        ubar_n.x.array[:ubar_offset -
                       p_offset] = x.array_r[p_offset:ubar_offset]
        ubar_n.x.scatter_forward()
        pbar_h.x.array[:pbar_offset -
                       ubar_offset] = x.array_r[ubar_offset:pbar_offset]
        pbar_h.x.scatter_forward()
        A_h.x.array[:(len(x.array_r) - pbar_offset)
                    ] = x.array_r[pbar_offset:]
        A_h.x.scatter_forward()

        # Interpolate for visualisation
        B_vis.interpolate(B_expr)
        u_vis.interpolate(u_n)

        # Write to file
        for vis_file in vis_files:
            vis_file.write(t)

    for vis_file in vis_files:
        vis_file.close()

    # Compute divergence and jump errors
    e_div_u = norm_L2(msh.comm, div(u_n))
    e_jump_u = normal_jump_error(submesh_f, u_n)
    par_print(comm, f"e_div_u = {e_div_u}")
    par_print(comm, f"e_jump_u = {e_jump_u}")

    # TODO Remove
    par_print(comm, x.norm())


if __name__ == "__main__":
    # Simulation parameters
    solver_type = SolverType.NAVIER_STOKES
    n_x = 12  # Number of elements in the x-direction
    n_y = 6  # Number of elements in the y-direction
    n_z = 6  # Number of elements in the z-direction in the fluid
    n_s_y = 2  # Number of elements in the z-direction in the wall
    sigma_f = 2.5  # Fluid conductivity
    sigma_s = 100  # Solid conductivity
    mu = 0.4  # Permeability
    k = 1  # Polynomial degree
    cell_type = mesh.CellType.hexahedron
    nu = 1.0e-3  # Kinematic viscosity
    num_time_steps = 10
    t_end = 10
    delta_t = t_end / num_time_steps
    scheme = Scheme.DRW
    comm = MPI.COMM_WORLD

    # Volume and boundary ids
    volumes = {"solid": 1,
               "fluid": 2}
    boundaries = {"solid_x_walls": 1,
                  "solid_y_walls": 2,
                  "solid_z_walls": 3,
                  "fluid_y_walls": 4,
                  "fluid_z_walls": 5,
                  "inlet": 6,
                  "outlet": 7}

    # Create the mesh
    gmsh.initialize()
    if comm.rank == 0:
        # Define geometry
        gmsh.model.add("channel")
        order = 1

        # Fluid domain x, y, and z lengths
        L_x = 10
        L_y = 1
        L_z = 1
        wall_thickness = 0.1
        domain_points = [gmsh.model.geo.addPoint(0, - wall_thickness, 0),
                         gmsh.model.geo.addPoint(L_x, - wall_thickness, 0),
                         gmsh.model.geo.addPoint(
                                L_x, L_y + wall_thickness, 0),
                         gmsh.model.geo.addPoint(
            0, L_y + wall_thickness, 0)]
        fluid_points = [gmsh.model.geo.addPoint(0, 0, 0),
                        gmsh.model.geo.addPoint(L_x, 0, 0),
                        gmsh.model.geo.addPoint(L_x, L_y, 0),
                        gmsh.model.geo.addPoint(0, L_y, 0)]

        wall_0_lines = [gmsh.model.geo.addLine(
            domain_points[0], domain_points[1]),
            gmsh.model.geo.addLine(
            domain_points[1], fluid_points[1]),
            gmsh.model.geo.addLine(
            fluid_points[1], fluid_points[0]),
            gmsh.model.geo.addLine(
            fluid_points[0], domain_points[0]),
        ]
        fluid_lines = [- wall_0_lines[2],
                       gmsh.model.geo.addLine(
                            fluid_points[1], fluid_points[2]),
                       gmsh.model.geo.addLine(
                            fluid_points[2], fluid_points[3]),
                       gmsh.model.geo.addLine(
            fluid_points[3], fluid_points[0])]

        wall_1_lines = [- fluid_lines[2],
                        gmsh.model.geo.addLine(
                            fluid_points[2], domain_points[2]),
                        gmsh.model.geo.addLine(
                            domain_points[2], domain_points[3]),
                        gmsh.model.geo.addLine(
            domain_points[3], fluid_points[3])]

        wall_0_loop = gmsh.model.geo.addCurveLoop(wall_0_lines)
        fluid_loop = gmsh.model.geo.addCurveLoop(fluid_lines)
        wall_1_loop = gmsh.model.geo.addCurveLoop(wall_1_lines)

        wall_0_surf = gmsh.model.geo.addPlaneSurface([wall_0_loop])
        fluid_surf = gmsh.model.geo.addPlaneSurface([fluid_loop])
        wall_1_surf = gmsh.model.geo.addPlaneSurface([wall_1_loop])

        gmsh.model.geo.mesh.setTransfiniteCurve(fluid_lines[0], n_x + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(fluid_lines[1], n_y + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(fluid_lines[2], n_x + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(fluid_lines[3], n_y + 1)
        gmsh.model.geo.mesh.setTransfiniteSurface(fluid_surf, "Left")

        gmsh.model.geo.mesh.setTransfiniteCurve(wall_0_lines[0], n_x + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(wall_0_lines[1], n_s_y + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(wall_0_lines[2], n_x + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(wall_0_lines[3], n_s_y + 1)
        gmsh.model.geo.mesh.setTransfiniteSurface(wall_0_surf, "Left")

        gmsh.model.geo.mesh.setTransfiniteCurve(wall_1_lines[0], n_x + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(wall_1_lines[1], n_s_y + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(wall_1_lines[2], n_x + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(wall_1_lines[3], n_s_y + 1)
        gmsh.model.geo.mesh.setTransfiniteSurface(wall_1_surf, "Left")

        if cell_type == mesh.CellType.tetrahedron:
            recombine = False
        else:
            recombine = True
        extrude_surfs = [(2, wall_0_surf),
                         (2, fluid_surf), (2, wall_1_surf)]
        gmsh.model.geo.extrude(
            extrude_surfs, 0, 0, L_z, [n_z], recombine=recombine)

        gmsh.model.geo.synchronize()

        # Physical groups
        gmsh.model.addPhysicalGroup(3, [1, 3], volumes["solid"])
        gmsh.model.addPhysicalGroup(3, [2], volumes["fluid"])

        gmsh.model.addPhysicalGroup(
            2, [23, 31, 67, 75], boundaries["solid_x_walls"])
        gmsh.model.addPhysicalGroup(
            2, [19, 71], boundaries["solid_y_walls"])
        gmsh.model.addPhysicalGroup(
            2, [1, 3, 32, 76], boundaries["solid_z_walls"])
        gmsh.model.addPhysicalGroup(
            2, [27, 49], boundaries["fluid_y_walls"])
        gmsh.model.addPhysicalGroup(
            2, [2, 54], boundaries["fluid_z_walls"])
        gmsh.model.addPhysicalGroup(2, [53], boundaries["inlet"])
        gmsh.model.addPhysicalGroup(2, [45], boundaries["outlet"])

        # gmsh.option.setNumber("Mesh.Smoothing", 5)
        if cell_type == mesh.CellType.quadrilateral \
                or cell_type == mesh.CellType.hexahedron:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            # TODO Check what this is doing, it may be making things worse
            # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(order)

        # gmsh.write("msh.msh")

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    msh, ct, ft = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=3, partitioner=partitioner)
    gmsh.finalize()

    # Fluid BCs
    def inlet(x): return np.vstack(
            (36 * x[1] * (1 - x[1]) * x[2] * (1 - x[2]),
             np.zeros_like(x[0]),
             np.zeros_like(x[0])))

    def zero(x): return np.vstack(
        (np.zeros_like(x[0]),
            np.zeros_like(x[0]),
            np.zeros_like(x[0])))
    u_bcs = {"inlet": (BCType.Dirichlet, inlet),
             "outlet": (BCType.Neumann, zero),
             "fluid_y_walls": (BCType.Dirichlet, zero),
             "fluid_z_walls": (BCType.Dirichlet, zero)}
    # Electromagnetic BCs
    # # Homogeneous Dirichlet (conducting) on y walls,
    # # homogeneous Neumann (insulating) on z walls
    # A_bcs = {"solid_y_walls": (BCType.Dirichlet, zero)}
    # Insulating walls
    A_bcs = {}

    boundary_conditions = {"u": u_bcs, "A": A_bcs}

    # Initial condition
    def u_i_expr(x): return np.zeros_like(x[:3])

    # Externally applied force on the fluid
    f = fem.Constant(msh, [PETSc.ScalarType(0.0) for i in range(3)])

    solve(solver_type, k, nu, num_time_steps, delta_t, scheme, msh, ct,
          ft, volumes, boundaries, boundary_conditions, f, u_i_expr, sigma_s,
          sigma_f, mu)
