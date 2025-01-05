# TODO FIXME SYMMETRIC GRADIENT OBJECTIVITY

# TODO Tidy up problems

# This demo solves the Stokes and Navier-Stokes equations using
# hybridised discontinuous Galerkin methods. There are two schemes
# to choose from:
#   1) the scheme presented in "Hybridized discontinuous Galerkin
#      methods for incompressible flows on meshes with quadrilateral
#      cells" by J. P. Dean, S. Rhebergen, and G. N. Well.
#   2) The scheme from "An embedded–hybridized discontinuous Galerkin
#      finite element method for the Stokes equations" by S. Rhebergen
#      and G. N. Wells

from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div, outer
import numpy as np
from petsc4py import PETSc
from utils import (
    norm_L2,
    domain_average,
    normal_jump_error,
    TimeDependentExpression,
    par_print,
    compute_cell_boundary_int_entities,
    markers_to_meshtags,
)
from enum import Enum
import gmsh
from dolfinx.io import gmshio
from dolfinx.fem.petsc import (
    assemble_matrix_block,
    assemble_vector_block,
    create_matrix_block,
)


class SolverType(Enum):
    STOKES = 1
    NAVIER_STOKES = 2


class Scheme(Enum):
    # The scheme by Dean, Rhebergen, and Wells
    DRW = 1
    # The scheme by Rhebergen and Wells
    RW = 2


class BCType(Enum):
    Dirichlet = 1
    Neumann = 2


def create_facet_mesh(msh):
    # Create a sub-mesh containing all of the facets in msh
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    facet_imap = msh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    facets = np.arange(num_facets, dtype=np.int32)

    # NOTE Despite all facets being present in the submesh, the entity
    # map isn't necessarily the identity in parallel
    facet_mesh, facet_mesh_to_msh = mesh.create_submesh(msh, fdim, facets)[0:2]

    return facet_mesh, facet_mesh_to_msh


def create_function_spaces(msh, facet_mesh, scheme, k):
    # Create function spaces
    if scheme == Scheme.RW:
        V = fem.functionspace(msh, ("Discontinuous Lagrange", k, (msh.geometry.dim,)))
        Q = fem.functionspace(msh, ("Discontinuous Lagrange", k - 1))
    else:
        V = fem.functionspace(msh, ("Discontinuous Raviart-Thomas", k + 1))
        Q = fem.functionspace(msh, ("Discontinuous Lagrange", k))
    Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k, (msh.geometry.dim,)))
    Qbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k))

    return V, Q, Vbar, Qbar


def create_forms(
    V,
    Q,
    Vbar,
    Qbar,
    msh,
    k,
    delta_t,
    nu,
    facet_mesh_to_msh,
    solver_type,
    boundary_conditions,
    boundaries,
    mt,
    f,
    facet_mesh,
    u_n,
    ubar_n,
):
    tdim = msh.topology.dim
    fdim = tdim - 1

    # We wish to integrate around the boundary of each cell, so we
    # get a list of cell boundary facets as follows:
    cell_boundary_facets = compute_cell_boundary_int_entities(msh)
    cell_boundaries_tag = 0
    # Add cell boundaries to the list of integration entities
    facet_integration_entities = [(cell_boundaries_tag, cell_boundary_facets)]
    # We also need to integrate over portions of the boundary (e.g. to
    # apply boundary conditions). We can add the required integration
    # entities as follows:
    facet_integration_entities += [
        (
            tag,
            fem.compute_integration_domains(
                fem.IntegralType.exterior_facet, msh.topology, mt.find(tag)
            ),
        )
        for tag in np.sort(list(boundaries.values()))
    ]

    # Create integration measures
    dx_c = ufl.Measure("dx", domain=msh)
    # FIXME This is being estimated wrong for DRW. Compute correctly!
    quad_deg = (k + 1) ** 2
    ds_c = ufl.Measure(
        "ds",
        subdomain_data=facet_integration_entities,
        domain=msh,
        metadata={"quadrature_degree": quad_deg},
    )

    # We write the mixed domain forms as integrals over msh. Hence, we must
    # provide a map from facets in msh to cells in facet_mesh. This is the
    # 'inverse' of facet_mesh_to_mesh, which we compute as follows:
    facet_imap = msh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    msh_to_facet_mesh = np.full(num_facets, -1)
    msh_to_facet_mesh[facet_mesh_to_msh] = np.arange(len(facet_mesh_to_msh))
    entity_maps = {facet_mesh: msh_to_facet_mesh}

    # Define trial and test functitons
    W = ufl.MixedFunctionSpace(V, Q, Vbar, Qbar)

    u, p, ubar, pbar = ufl.TrialFunctions(W)
    v, q, vbar, qbar = ufl.TestFunctions(W)

    h = ufl.CellDiameter(msh)  # TODO Fix for high-order geom!
    n = ufl.FacetNormal(msh)
    # Scaled penalty parameter. TODO Should be larger in 3D!
    gamma = 16 * k**2 / h

    # Marker for outflow boundaries
    lmbda = ufl.conditional(ufl.lt(dot(u_n, n), 0), 1, 0)

    # Convert some parameters to constants
    delta_t = fem.Constant(msh, PETSc.ScalarType(delta_t))
    nu = fem.Constant(msh, PETSc.ScalarType(nu))

    # NOTE: On domain boundary, ubar terms with Dirichlet BC applied and
    # non-zero test function are effectively moved to the RHS by apply_lifting
    a = (
        inner(u / delta_t, v) * dx_c
        + nu * inner(grad(u), grad(v)) * dx_c
        - nu * inner(u - ubar, dot(grad(v), n)) * ds_c(cell_boundaries_tag)
        - nu * inner(dot(grad(u), n), v - vbar) * ds_c(cell_boundaries_tag)
        + nu * gamma * inner(u - ubar, v - vbar) * ds_c(cell_boundaries_tag)
        - inner(p, div(v)) * dx_c
        + inner(pbar, dot(v, n)) * ds_c(cell_boundaries_tag)
        - inner(div(u), q) * dx_c
        + inner(dot(u, n), qbar) * ds_c(cell_boundaries_tag)
    )

    # Advective terms
    if solver_type == SolverType.NAVIER_STOKES:
        a += -inner(outer(u, u_n), grad(v)) * dx_c + inner(
            outer(u, u_n) - outer((u - ubar), lmbda * u_n), outer((v - vbar), n)
        ) * ds_c(cell_boundaries_tag)

    L = inner(f + u_n / delta_t, v) * dx_c
    L += inner(fem.Constant(msh, [PETSc.ScalarType(0.0) for i in range(tdim)]), vbar) * ds_c(
        cell_boundaries_tag
    )
    L += inner(fem.Constant(facet_mesh, PETSc.ScalarType(0.0)), qbar) * ds_c(cell_boundaries_tag)
    L += inner(fem.Constant(msh, 0.0), q) * dx_c

    # Apply boundary conditions
    bcs = []
    # FIXME Can now access the bc_func through the bc
    bc_funcs = []
    facet_mesh.topology.create_connectivity(fdim, fdim)
    for name, bc in boundary_conditions.items():
        id = boundaries[name]
        bc_type, bc_expr = bc
        if bc_type == BCType.Dirichlet:
            bc_func = fem.Function(Vbar)
            bc_func.interpolate(bc_expr)
            bc_funcs.append((bc_func, bc_expr))
            facets = msh_to_facet_mesh[mt.indices[mt.values == id]]
            dofs = fem.locate_dofs_topological(Vbar, fdim, facets)
            bcs.append(fem.dirichletbc(bc_func, dofs))
            L += inner(dot(bc_func, n), qbar) * ds_c(id)
        else:
            assert bc_type == BCType.Neumann
            L += -inner(bc_expr, vbar) * ds_c(id)
            a += -inner(dot(ubar, n), qbar) * ds_c(id) - inner(dot(vbar, n), pbar) * ds_c(id)
            if solver_type == SolverType.NAVIER_STOKES:
                a += inner((1 - lmbda) * dot(ubar_n, n) * ubar, vbar) * ds_c(id)

    # Compile LHS forms
    a = fem.form(ufl.extract_blocks(a), entity_maps=entity_maps)
    L = fem.form(ufl.extract_blocks(L), entity_maps=entity_maps)

    return a, L, bcs, bc_funcs


def compute_offsets(V, Q, Vbar):
    u_offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    p_offset = u_offset + Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    ubar_offset = p_offset + Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
    return u_offset, p_offset, ubar_offset


def solve(
    solver_type,
    k,
    nu,
    num_time_steps,
    delta_t,
    scheme,
    msh,
    mt,
    boundaries,
    boundary_conditions,
    f,
    u_i_expr,
    u_e=None,
    p_e=None,
):
    comm = msh.comm

    # Create a mesh containing the facets of the mesh
    facet_mesh, entity_map = create_facet_mesh(msh)

    # Create function spaces
    V, Q, Vbar, Qbar = create_function_spaces(msh, facet_mesh, scheme, k)

    # Cell velocity at previous time step
    u_n = fem.Function(V)
    # Facet velocity at previous time step
    ubar_n = fem.Function(Vbar)

    # Interpolate initial condition
    u_n.interpolate(u_i_expr)
    ubar_n.interpolate(u_i_expr)

    # Create finite element forms
    a, L, bcs, bc_funcs = create_forms(
        V,
        Q,
        Vbar,
        Qbar,
        msh,
        k,
        delta_t,
        nu,
        entity_map,
        solver_type,
        boundary_conditions,
        boundaries,
        mt,
        f,
        facet_mesh,
        u_n,
        ubar_n,
    )

    # Set up matrix
    if solver_type == SolverType.NAVIER_STOKES:
        A = create_matrix_block(a)
    else:
        A = assemble_matrix_block(a, bcs=bcs)
        A.assemble()

    # Create vectors for RHS and solution
    b = fem.petsc.create_vector_block(L)
    x = A.createVecRight()

    # Configure solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts[
        "mat_mumps_icntl_24"
    ] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts[
        "mat_mumps_icntl_25"
    ] = 0  # Option to support solving a singular matrix (pressure nullspace)
    ksp.setFromOptions()

    # Prepare functions for visualisation
    # Cell velocity
    if scheme == Scheme.RW:
        u_vis = fem.Function(V)
    else:
        # Since the DRW scheme uses a broken RT space for the velocity,
        # we create a discontinuous Lagrange space to interpolate the
        # solution into for visualisation. This allows artifact-free
        # visualisation of the solution
        V_vis = fem.functionspace(msh, ("Discontinuous Lagrange", k + 1, (msh.geometry.dim,)))
        u_vis = fem.Function(V_vis)
    u_vis.name = "u"
    u_vis.interpolate(u_n)
    # Cell pressure
    p_h = fem.Function(Q)
    p_h.name = "p"
    # Facet pressure
    pbar_h = fem.Function(Qbar)
    pbar_h.name = "pbar"

    # Set up files for visualisation
    vis_files = [
        io.VTXWriter(msh.comm, file_name, [func._cpp_object], "BP4")
        for (file_name, func) in [
            ("u.bp", u_vis),
            ("p.bp", p_h),
            ("ubar.bp", ubar_n),
            ("pbar.bp", pbar_h),
        ]
    ]

    # Time stepping loop
    t = 0.0
    u_offset, p_offset, ubar_offset = compute_offsets(V, Q, Vbar)
    for vis_file in vis_files:
        vis_file.write(t)
    for n in range(num_time_steps):
        t += delta_t
        par_print(comm, f"t = {t}")

        # Update any boundary data
        for bc_func, bc_expr in bc_funcs:
            if isinstance(bc_expr, TimeDependentExpression):
                bc_expr.t = t
                bc_func.interpolate(bc_expr)

        # Assemble LHS
        if solver_type == SolverType.NAVIER_STOKES:
            A.zeroEntries()
            assemble_matrix_block(A, a, bcs=bcs)
            A.assemble()

        # Assemble RHS
        with b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector_block(b, L, a, bcs=bcs)

        # Compute solution
        ksp.solve(b, x)

        # Recover solution
        u_n.x.array[:u_offset] = x.array_r[:u_offset]
        u_n.x.scatter_forward()
        p_h.x.array[: p_offset - u_offset] = x.array_r[u_offset:p_offset]
        p_h.x.scatter_forward()
        ubar_n.x.array[: ubar_offset - p_offset] = x.array_r[p_offset:ubar_offset]
        ubar_n.x.scatter_forward()
        pbar_h.x.array[: (len(x.array_r) - ubar_offset)] = x.array_r[ubar_offset:]
        pbar_h.x.scatter_forward()

        # Interpolate solution for visualisation
        u_vis.interpolate(u_n)

        # Write to file
        for vis_file in vis_files:
            vis_file.write(t)

    for vis_file in vis_files:
        vis_file.close()

    # Compute errors
    e_div_u = norm_L2(msh.comm, div(u_n))
    par_print(comm, f"e_div_u = {e_div_u}")

    # FIXME: Parallel (due to mesh ghost mode none)
    if comm.size == 1:
        e_jump_u = normal_jump_error(msh, u_n)
        par_print(comm, f"e_jump_u = {e_jump_u}")

    x = ufl.SpatialCoordinate(msh)
    xbar = ufl.SpatialCoordinate(facet_mesh)
    if u_e is not None:
        e_u = norm_L2(msh.comm, u_n - u_e(x))
        e_ubar = norm_L2(msh.comm, ubar_n - u_e(xbar))
        par_print(comm, f"e_u = {e_u}")
        par_print(comm, f"e_ubar = {e_ubar}")

    if p_e is not None:
        p_h_avg = domain_average(msh, p_h)
        p_e_avg = domain_average(msh, p_e(x))
        e_p = norm_L2(msh.comm, (p_h - p_h_avg) - (p_e(x) - p_e_avg))
        pbar_h_avg = domain_average(facet_mesh, pbar_h)
        pbar_e_avg = domain_average(facet_mesh, p_e(xbar))
        e_pbar = norm_L2(msh.comm, (pbar_h - pbar_h_avg) - (p_e(xbar) - pbar_e_avg))

        par_print(comm, f"e_p = {e_p}")
        par_print(comm, f"e_pbar = {e_pbar}")


def run_square_problem():
    # Simulation parameters
    comm = MPI.COMM_WORLD
    scheme = Scheme.DRW
    solver_type = SolverType.NAVIER_STOKES
    h = 1 / 16  # Maximum cell diameter
    k = 3  # Polynomial degree
    cell_type = mesh.CellType.quadrilateral
    nu = 1.0e-6  # Kinematic viscosity
    num_time_steps = 320
    t_end = 40
    d = 2

    # Create mesh
    n = round(1 / h)
    if d == 2:
        msh = mesh.create_unit_square(comm, n, n, cell_type, ghost_mode=mesh.GhostMode.none)

        def diri_boundary_marker(x):
            return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

        def neumann_boundary_marker(x):
            return np.isclose(x[1], 1.0)
    else:
        msh = mesh.create_unit_cube(comm, n, n, n, cell_type, mesh.GhostMode.none)

        def boundary_marker(x):
            return (
                np.isclose(x[0], 0.0)
                | np.isclose(x[0], 1.0)
                | np.isclose(x[1], 0.0)
                | np.isclose(x[1], 1.0)
                | np.isclose(x[2], 0.0)
                | np.isclose(x[2], 1.0)
            )

    # Create meshtags for boundary
    fdim = msh.topology.dim - 1
    boundaries = {"dirichlet": 1, "neumann": 2}
    markers = [diri_boundary_marker, neumann_boundary_marker]
    mt = markers_to_meshtags(msh, boundaries.values(), markers, fdim)

    # Exact velocity
    def u_e(x, module=ufl):
        if d == 2:
            u = (
                module.sin(module.pi * x[0]) * module.sin(module.pi * x[1]),
                module.cos(module.pi * x[0]) * module.cos(module.pi * x[1]),
            )
        else:
            u = (
                module.sin(module.pi * x[0]) * module.cos(module.pi * x[1])
                - module.sin(module.pi * x[0]) * module.cos(module.pi * x[2]),
                module.sin(module.pi * x[1]) * module.cos(module.pi * x[2])
                - module.sin(module.pi * x[1]) * module.cos(module.pi * x[0]),
                module.sin(module.pi * x[2]) * module.cos(module.pi * x[0])
                - module.sin(module.pi * x[2]) * module.cos(module.pi * x[1]),
            )
        if module == ufl:
            return ufl.as_vector(u)
        else:
            assert module == np
            return np.vstack(u)

    # Exact pressure
    def p_e(x, module=ufl):
        if d == 2:
            return module.sin(module.pi * x[0]) * module.cos(module.pi * x[1])
        else:
            return (
                module.sin(module.pi * x[0])
                * module.cos(module.pi * x[1])
                * module.sin(module.pi * x[2])
            )

    # Right-hand side
    x = ufl.SpatialCoordinate(msh)
    sigma = p_e(x) * ufl.Identity(msh.topology.dim) - nu * grad(u_e(x))
    if solver_type == SolverType.NAVIER_STOKES:
        sigma += outer(u_e(x), u_e(x))
    f = div(sigma)

    n = ufl.FacetNormal(msh)
    g = dot(sigma, n)
    if solver_type == SolverType.NAVIER_STOKES:
        g += -ufl.conditional(ufl.gt(dot(u_e(x), n), 0), dot(u_e(x), n), 0) * u_e(x)

    # Boundary conditions
    boundary_conditions = {
        "dirichlet": (BCType.Dirichlet, lambda x: u_e(x, module=np)),
        "neumann": (BCType.Neumann, g),
    }

    # Initial condition
    def u_i(x):
        return np.zeros_like(x[:d])

    # Call solver
    delta_t = t_end / num_time_steps
    solve(
        solver_type,
        k,
        nu,
        num_time_steps,
        delta_t,
        scheme,
        msh,
        mt,
        boundaries,
        boundary_conditions,
        f,
        u_i,
        u_e,
        p_e,
    )


def run_gaussian_bump():
    # Simulation parameters
    comm = MPI.COMM_WORLD
    scheme = Scheme.DRW
    solver_type = SolverType.NAVIER_STOKES
    h = 1 / 16  # Maximum cell diameter
    k = 3  # Polynomial degree
    cell_type = mesh.CellType.quadrilateral
    nu = 1.0e-3  # Kinematic viscosity
    num_time_steps = 10
    t_end = 10

    # Create mesh
    def gaussian(x, a, sigma, mu):
        return a * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)

    # Create the mesh
    gmsh.initialize()
    if comm.rank == 0:
        # TODO Pass options
        gmsh.model.add("gaussian_bump")
        a = 0.2
        sigma = 0.2
        mu = 1.0
        w = 5.0
        order = 1
        num_bottom_points = 100

        # Point tags
        bottom_points = [
            gmsh.model.geo.addPoint(x, gaussian(x, a, sigma, mu), 0.0, h)
            for x in np.linspace(0.0, w, num_bottom_points)
        ]
        top_left_point = gmsh.model.geo.addPoint(0, 1, 0, h)
        top_right_point = gmsh.model.geo.addPoint(w, 1, 0, h)

        # Line tags
        lines = []
        lines.append(gmsh.model.geo.addSpline(bottom_points))
        lines.append(gmsh.model.geo.addLine(bottom_points[-1], top_right_point))
        lines.append(gmsh.model.geo.addLine(top_right_point, top_left_point))
        lines.append(gmsh.model.geo.addLine(top_left_point, bottom_points[0]))

        gmsh.model.geo.addCurveLoop(lines, 1)

        gmsh.model.geo.addPlaneSurface([1], 1)

        # gmsh.model.geo.mesh.setTransfiniteCurve(1, 40)
        # gmsh.model.geo.mesh.setTransfiniteCurve(
        #   2, 15, "Progression", 1.1)
        # gmsh.model.geo.mesh.setTransfiniteCurve(3, 40)
        # gmsh.model.geo.mesh.setTransfiniteCurve(
        #   4, 15, "Progression", -1.1)
        # gmsh.model.geo.mesh.setTransfiniteSurface(
        #     1, "Left", [bottom_points[0], bottom_points[-1],
        #                 top_right_point, top_left_point])

        gmsh.model.geo.synchronize()

        # Add physical groups
        gmsh.model.addPhysicalGroup(2, [1], 1)

        gmsh.model.addPhysicalGroup(1, [lines[0]], 1)
        gmsh.model.addPhysicalGroup(1, [lines[1]], 2)
        gmsh.model.addPhysicalGroup(1, [lines[2]], 3)
        gmsh.model.addPhysicalGroup(1, [lines[3]], 4)

        gmsh.option.setNumber("Mesh.Smoothing", 5)
        if cell_type == mesh.CellType.quadrilateral:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            # TODO Check what this is doing, it may be making things worse
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    msh, _, mt = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2, partitioner=partitioner)
    gmsh.finalize()
    boundaries = {"left": 4, "bottom": 1, "top": 3, "right": 2}

    # Boundary conditions
    def inlet(x):
        return np.vstack((np.ones_like(x[0]), np.zeros_like(x[0])))

    def zero(x):
        return np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0])))

    boundary_conditions = {
        "left": (BCType.Dirichlet, inlet),
        "right": (BCType.Neumann, ufl.as_vector((1e-12, 0.0))),
        "bottom": (BCType.Dirichlet, zero),
        "top": (BCType.Dirichlet, zero),
    }

    # Right-hand side
    f = fem.Constant(msh, (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0)))

    # Initial condition
    def u_i(x):
        return np.zeros_like(x[:2])

    # Call solver
    delta_t = t_end / num_time_steps
    solve(
        solver_type,
        k,
        nu,
        num_time_steps,
        delta_t,
        scheme,
        msh,
        mt,
        boundaries,
        boundary_conditions,
        f,
        u_i,
    )


def run_cylinder_problem():
    # Simulation parameters
    comm = MPI.COMM_WORLD
    scheme = Scheme.DRW
    solver_type = SolverType.NAVIER_STOKES
    h = 1 / 24  # Maximum cell diameter
    k = 3  # Polynomial degree
    cell_type = mesh.CellType.quadrilateral
    nu = 1.0e-3  # Kinematic viscosity
    num_time_steps = 100
    t_end = 1
    d = 2

    # Volume and boundary ids
    volume_id = {"fluid": 1}
    boundary_id = {"inlet": 2, "outlet": 3, "wall": 4, "obstacle": 5}

    # Create mesh
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("model")
        factory = gmsh.model.geo

        if d == 2:
            length = 2.2
            c = (0.2, 0.2)
        else:
            length = 2.5
            c = (0.5, 0.2)
        height = 0.41
        r = 0.05
        r_s = 0.15
        order = 1

        rectangle_points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(length, 0.0, 0.0, h),
            factory.addPoint(length, height, 0.0, h),
            factory.addPoint(0.0, height, 0.0, h),
        ]

        thetas = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4, 9 * np.pi / 4]
        circle_points = [factory.addPoint(c[0], c[1], 0.0)] + [
            factory.addPoint(c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), 0.0)
            for theta in thetas
        ]

        square_points = [
            factory.addPoint(c[0] + r_s * np.cos(theta), c[1] + r_s * np.sin(theta), 0.0)
            for theta in thetas
        ]

        rectangle_lines = [
            factory.addLine(rectangle_points[0], rectangle_points[1]),
            factory.addLine(rectangle_points[1], rectangle_points[2]),
            factory.addLine(rectangle_points[2], rectangle_points[3]),
            factory.addLine(rectangle_points[3], rectangle_points[0]),
        ]

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

        bl_diag_lines = [factory.addLine(circle_points[i + 1], square_points[i]) for i in range(4)]

        boundary_layer_lines = [
            [square_lines[0], -bl_diag_lines[1], -circle_lines[0], bl_diag_lines[0]],
            [square_lines[1], -bl_diag_lines[2], -circle_lines[1], bl_diag_lines[1]],
            [square_lines[2], -bl_diag_lines[3], -circle_lines[2], bl_diag_lines[2]],
            [square_lines[3], -bl_diag_lines[0], -circle_lines[3], bl_diag_lines[3]],
        ]

        rectangle_curve = factory.addCurveLoop(rectangle_lines)
        factory.addCurveLoop(circle_lines)
        square_curve = factory.addCurveLoop(square_lines)
        boundary_layer_curves = [factory.addCurveLoop(bll) for bll in boundary_layer_lines]

        outer_surface = factory.addPlaneSurface([rectangle_curve, square_curve])
        boundary_layer_surfaces = [factory.addPlaneSurface([blc]) for blc in boundary_layer_curves]

        num_bl_eles = round(0.5 * 1 / h)
        progression_coeff = 1.2
        for i in range(len(boundary_layer_surfaces)):
            gmsh.model.geo.mesh.setTransfiniteCurve(boundary_layer_lines[i][0], num_bl_eles)
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][1], num_bl_eles, coef=progression_coeff
            )
            gmsh.model.geo.mesh.setTransfiniteCurve(boundary_layer_lines[i][2], num_bl_eles)
            gmsh.model.geo.mesh.setTransfiniteCurve(
                boundary_layer_lines[i][3], num_bl_eles, coef=progression_coeff
            )
            gmsh.model.geo.mesh.setTransfiniteSurface(boundary_layer_surfaces[i])

        if d == 3:
            if cell_type == mesh.CellType.tetrahedron:
                recombine = False
            else:
                recombine = True
            extrude_surfs = [(2, surf) for surf in [outer_surface] + boundary_layer_surfaces]
            gmsh.model.geo.extrude(extrude_surfs, 0, 0, 0.41, [8], recombine=recombine)

        gmsh.model.geo.synchronize()

        # Add physical groups
        if d == 2:
            gmsh.model.addPhysicalGroup(
                2, [outer_surface] + boundary_layer_surfaces, volume_id["fluid"]
            )

            gmsh.model.addPhysicalGroup(
                1, [rectangle_lines[0], rectangle_lines[2]], boundary_id["wall"]
            )
            gmsh.model.addPhysicalGroup(1, [rectangle_lines[1]], boundary_id["outlet"])
            gmsh.model.addPhysicalGroup(1, [rectangle_lines[3]], boundary_id["inlet"])
            gmsh.model.addPhysicalGroup(1, circle_lines, boundary_id["obstacle"])
        else:
            # FIXME Mark without hardcoding
            gmsh.model.addPhysicalGroup(3, [1, 2, 3, 4, 5], volume_id["fluid"])

            gmsh.model.addPhysicalGroup(2, [41], boundary_id["inlet"])

            gmsh.model.addPhysicalGroup(2, [33], boundary_id["outlet"])

            gmsh.model.addPhysicalGroup(
                2, [1, 2, 3, 4, 5, 29, 37, 58, 80, 102, 124, 146], boundary_id["wall"]
            )

            gmsh.model.addPhysicalGroup(2, [75, 97, 119, 141], boundary_id["obstacle"])

        # gmsh.option.setNumber("Mesh.Smoothing", 5)
        if cell_type == mesh.CellType.quadrilateral or cell_type == mesh.CellType.hexahedron:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.model.mesh.generate(d)
        gmsh.model.mesh.setOrder(order)

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    msh, _, mt = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=d, partitioner=partitioner)
    gmsh.finalize()

    # Boundary conditions
    if d == 2:

        def inlet(x):
            return np.vstack(((1.5 * 4 * x[1] * (0.41 - x[1])) / 0.41**2, np.zeros_like(x[0])))

        def zero(x):
            return np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0])))
    else:
        H = 0.41

        def inlet(x):
            return np.vstack(
                (
                    16 * 0.45 * x[1] * x[2] * (H - x[1]) * (H - x[2]) / H**4,
                    np.zeros_like(x[0]),
                    np.zeros_like(x[0]),
                )
            )

        def zero(x):
            return np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))

    boundary_conditions = {
        "inlet": (BCType.Dirichlet, inlet),
        "outlet": (BCType.Neumann, ufl.as_vector((1e-16, 0.0))),
        "wall": (BCType.Dirichlet, zero),
        "obstacle": (BCType.Dirichlet, zero),
    }

    # Forcing term
    f = fem.Constant(msh, [PETSc.ScalarType(0.0) for i in range(d)])

    # Initial condition
    def u_i(x):
        return np.zeros_like(x[:d])

    # Call solver
    delta_t = t_end / num_time_steps
    solve(
        solver_type,
        k,
        nu,
        num_time_steps,
        delta_t,
        scheme,
        msh,
        mt,
        boundary_id,
        boundary_conditions,
        f,
        u_i,
    )


def run_taylor_green_problem():
    # Simulation parameters
    comm = MPI.COMM_WORLD
    scheme = Scheme.DRW
    solver_type = SolverType.NAVIER_STOKES
    h = 1 / 32  # Maximum cell diameter
    k = 3  # Polynomial degree
    cell_type = mesh.CellType.quadrilateral
    nu = 1.0e-4  # Kinematic viscosity
    num_time_steps = 20
    t_end = 1000
    Re = 1 / nu

    # Create mesh
    n = round(1 / h)
    point_0 = (-np.pi / 2, -np.pi / 2)
    point_1 = (np.pi / 2, np.pi / 2)
    msh = mesh.create_rectangle(
        comm, (point_0, point_1), (n, n), cell_type, ghost_mode=mesh.GhostMode.none
    )

    fdim = msh.topology.dim - 1
    boundaries = {"boundary": 1}

    def boundary_marker(x):
        return (
            np.isclose(x[0], point_0[0])
            | np.isclose(x[0], point_1[0])
            | np.isclose(x[1], point_0[1])
            | np.isclose(x[1], point_1[1])
        )

    mt = markers_to_meshtags(msh, boundaries.values(), [boundary_marker], fdim)

    # Exact solution
    def u_expr(x, t, module):
        return (
            -module.cos(x[0]) * module.sin(x[1]) * module.exp(-2 * t / Re),
            module.sin(x[0]) * module.cos(x[1]) * module.exp(-2 * t / Re),
        )

    def u_e(x, module=ufl):
        return ufl.as_vector(u_expr(x, t_end, ufl))

    def p_e(x):
        return -1 / 4 * (ufl.cos(2 * x[0]) + ufl.cos(2 * x[1])) * ufl.exp(-4 * t_end / Re)

    # Boundary conditions
    boundary_conditions = {
        "boundary": (
            BCType.Dirichlet,
            TimeDependentExpression(lambda x, t: np.vstack(u_expr(x, t, np))),
        )
    }

    # Initial condition
    def u_i(x):
        return u_expr(x, t=0, module=np)

    # Forcing term
    f = ufl.as_vector((0.0, 0.0))

    # Call solver
    delta_t = t_end / num_time_steps
    solve(
        solver_type,
        k,
        nu,
        num_time_steps,
        delta_t,
        scheme,
        msh,
        mt,
        boundaries,
        boundary_conditions,
        f,
        u_i,
        u_e,
        p_e,
    )


def run_kovasznay_problem():
    # Simulation parameters
    comm = MPI.COMM_WORLD
    scheme = Scheme.DRW
    solver_type = SolverType.NAVIER_STOKES
    h = 1 / 16  # Maximum cell diameter
    k = 3  # Polynomial degree
    cell_type = mesh.CellType.triangle
    nu = 1e-5  # Kinematic viscosity
    num_time_steps = 128
    t_end = 120

    # Create mesh
    n = round(1 / h)
    point_0 = (0.0, -0.5)
    point_1 = (1, 1.5)
    msh = mesh.create_rectangle(
        comm, (point_0, point_1), (n, 2 * n), cell_type, ghost_mode=mesh.GhostMode.none
    )
    fdim = msh.topology.dim - 1
    boundaries = {"boundary": 1}

    def boundary_marker(x):
        return (
            np.isclose(x[0], point_0[0])
            | np.isclose(x[0], point_1[0])
            | np.isclose(x[1], point_0[1])
            | np.isclose(x[1], point_1[1])
        )

    mt = markers_to_meshtags(msh, boundaries.values(), [boundary_marker], fdim)

    # Reynold's number
    R_e = 1 / nu

    # Exact velocity
    def u_e(x, module=ufl):
        u_x = 1 - module.exp(
            (R_e / 2 - module.sqrt(R_e**2 / 4 + 4 * module.pi**2)) * x[0]
        ) * module.cos(2 * module.pi * x[1])
        u_y = (
            (R_e / 2 - module.sqrt(R_e**2 / 4 + 4 * module.pi**2))
            / (2 * module.pi)
            * module.exp((R_e / 2 - module.sqrt(R_e**2 / 4 + 4 * module.pi**2)) * x[0])
            * module.sin(2 * module.pi * x[1])
        )
        if module == ufl:
            return ufl.as_vector((u_x, u_y))
        else:
            assert module == np
            return np.vstack((u_x, u_y))

    # Exact pressure
    def p_e(x, module=ufl):
        return (1 / 2) * (
            1 - module.exp(2 * (R_e / 2 - module.sqrt(R_e**2 / 4 + 4 * module.pi**2)) * x[0])
        )

    # Boundary conditions
    boundary_conditions = {"boundary": (BCType.Dirichlet, lambda x: u_e(x, module=np))}

    # Forcing term
    f = fem.Constant(msh, (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0)))

    # Initial condition
    def u_i(x):
        return np.zeros_like(x[:2])

    # Call solver
    delta_t = t_end / num_time_steps
    solve(
        solver_type,
        k,
        nu,
        num_time_steps,
        delta_t,
        scheme,
        msh,
        mt,
        boundaries,
        boundary_conditions,
        f,
        u_i,
        u_e,
        p_e,
    )


if __name__ == "__main__":
    run_square_problem()
