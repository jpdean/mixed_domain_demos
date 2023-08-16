# FIXME This demo needs tidying

from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div, outer
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.cpp.fem import compute_integration_domains
from utils import (norm_L2, domain_average, normal_jump_error,
                   TimeDependentExpression)
from enum import Enum
import gmsh
from dolfinx.io import gmshio
import sys


class SolverType(Enum):
    STOKES = 1
    NAVIER_STOKES = 2


class Scheme(Enum):
    # Scheme from https://doi.org/10.1016/j.cma.2019.112619
    RW = 1
    # Scheme from "Hybridized discontinuous Galerkin methods
    # for incompressible flows on meshes with quadrilateral
    # cells" by J. P. Dean, S. Rhebergen, and G. N. Wells
    DRW = 2


class BCType(Enum):
    Dirichlet = 1
    Neumann = 2


def par_print(string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def create_facet_mesh(msh):
    tdim = msh.topology.dim
    fdim = tdim - 1

    msh.topology.create_entities(fdim)
    facet_imap = msh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    facets = np.arange(num_facets, dtype=np.int32)

    # NOTE Despite all facets being present in the submesh, the entity
    # map isn't necessarily the identity in parallel
    facet_mesh, entity_map = mesh.create_submesh(msh, fdim, facets)[0:2]

    return facet_mesh, entity_map


def create_function_spaces(msh, facet_mesh, scheme, k):
    if scheme == Scheme.RW:
        V = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k))
        Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k - 1))
    else:
        V = fem.FunctionSpace(msh, ("Discontinuous Raviart-Thomas", k + 1))
        Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
    Vbar = fem.VectorFunctionSpace(
        facet_mesh, ("Discontinuous Lagrange", k))
    Qbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

    return V, Q, Vbar, Qbar


def create_forms(V, Q, Vbar, Qbar, msh, k, delta_t, nu,
                 entity_map, solver_type, boundary_conditions,
                 boundaries, mt, f, facet_mesh, u_n, ubar_n):
    tdim = msh.topology.dim
    fdim = tdim - 1

    all_facets_tag = 0
    all_facets = []
    num_cell_facets = cell_num_entities(msh.topology.cell_type, fdim)
    for cell in range(msh.topology.index_map(tdim).size_local):
        for local_facet in range(num_cell_facets):
            all_facets.extend([cell, local_facet])

    facet_integration_entities = [(all_facets_tag, all_facets)]
    facet_integration_entities += compute_integration_domains(
        fem.IntegralType.exterior_facet, mt._cpp_object)
    dx_c = ufl.Measure("dx", domain=msh)
    # FIXME Figure out why this is being estimated wrong for DRW
    # NOTE k**2 works on affine meshes
    quad_deg = (k + 1)**2
    ds_c = ufl.Measure(
        "ds", subdomain_data=facet_integration_entities, domain=msh,
        metadata={"quadrature_degree": quad_deg})
    dx_f = ufl.Measure("dx", domain=facet_mesh)

    inv_entity_map = np.full_like(entity_map, -1)
    for i, facet in enumerate(entity_map):
        inv_entity_map[facet] = i

    entity_maps = {facet_mesh: inv_entity_map}

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    ubar = ufl.TrialFunction(Vbar)
    vbar = ufl.TestFunction(Vbar)
    pbar = ufl.TrialFunction(Qbar)
    qbar = ufl.TestFunction(Qbar)

    h = ufl.CellDiameter(msh)  # TODO Fix for high order geom!
    n = ufl.FacetNormal(msh)
    gamma = 256.0 * k**2 / h  # TODO Should be larger in 3D

    lmbda = ufl.conditional(ufl.lt(dot(u_n, n), 0), 1, 0)
    delta_t = fem.Constant(msh, PETSc.ScalarType(delta_t))
    nu = fem.Constant(msh, PETSc.ScalarType(nu))

    a_00 = inner(u / delta_t, v) * dx_c \
        + nu * inner(grad(u), grad(v)) * dx_c \
        - nu * inner(grad(u), outer(v, n)) * ds_c(all_facets_tag) \
        + nu * gamma * inner(outer(u, n), outer(v, n)) * ds_c(all_facets_tag) \
        - nu * inner(outer(u, n), grad(v)) * ds_c(all_facets_tag)
    a_01 = fem.form(- inner(p * ufl.Identity(msh.topology.dim),
                    grad(v)) * dx_c)
    a_02 = - nu * gamma * inner(
        outer(ubar, n), outer(v, n)) * ds_c(all_facets_tag) \
        + nu * inner(outer(ubar, n), grad(v)) * ds_c(all_facets_tag)
    a_03 = fem.form(inner(pbar * ufl.Identity(msh.topology.dim),
                          outer(v, n)) * ds_c(all_facets_tag),
                    entity_maps=entity_maps)
    a_10 = fem.form(inner(u, grad(q)) * dx_c -
                    inner(dot(u, n), q) * ds_c(all_facets_tag))
    a_20 = - nu * inner(grad(u), outer(vbar, n)) * ds_c(all_facets_tag) \
        + nu * gamma * inner(outer(u, n), outer(vbar, n)
                             ) * ds_c(all_facets_tag)
    a_30 = fem.form(inner(dot(u, n), qbar) *
                    ds_c(all_facets_tag), entity_maps=entity_maps)
    a_23 = fem.form(
        inner(pbar * ufl.Identity(tdim), outer(vbar, n)) *
        ds_c(all_facets_tag),
        entity_maps=entity_maps)
    # On the Dirichlet boundary, the contribution from this term will be
    # added to the RHS in apply_lifting
    a_32 = fem.form(- inner(dot(ubar, n), qbar) * ds_c,
                    entity_maps=entity_maps)
    a_22 = - nu * gamma * \
        inner(outer(ubar, n), outer(vbar, n)) * ds_c(all_facets_tag)

    if solver_type == SolverType.NAVIER_STOKES:
        a_00 += - inner(outer(u, u_n), grad(v)) * dx_c \
            + inner(outer(u, u_n), outer(v, n)) * ds_c(all_facets_tag) \
            - inner(outer(u, lmbda * u_n), outer(v, n)) * ds_c(all_facets_tag)
        a_02 += inner(outer(ubar, lmbda * u_n), outer(v, n)) * \
            ds_c(all_facets_tag)
        a_20 += inner(outer(u, u_n), outer(vbar, n)) * ds_c(all_facets_tag) \
            - inner(outer(u, lmbda * u_n), outer(vbar, n)) * \
            ds_c(all_facets_tag)
        a_22 += inner(outer(ubar, lmbda * u_n),
                      outer(vbar, n)) * ds_c(all_facets_tag)

    L_2 = inner(fem.Constant(msh, [PETSc.ScalarType(0.0)
                                   for i in range(tdim)]),
                vbar) * ds_c(all_facets_tag)

    # NOTE: Don't set pressure BC to avoid affecting conservation properties.
    # MUMPS seems to cope with the small nullspace
    bcs = []
    bc_funcs = []
    for name, bc in boundary_conditions.items():
        id = boundaries[name]
        bc_type, bc_expr = bc
        bc_func = fem.Function(Vbar)
        bc_func.interpolate(bc_expr)
        bc_funcs.append((bc_func, bc_expr))
        if bc_type == BCType.Dirichlet:
            facets = inv_entity_map[mt.indices[mt.values == id]]
            dofs = fem.locate_dofs_topological(Vbar, fdim, facets)
            bcs.append(fem.dirichletbc(bc_func, dofs))
        else:
            assert bc_type == BCType.Neumann
            L_2 += inner(bc_func, vbar) * ds_c(id)
            if solver_type == SolverType.NAVIER_STOKES:
                a_22 += - inner((1 - lmbda) * dot(ubar_n, n) *
                                ubar, vbar) * ds_c(id)

    a_00 = fem.form(a_00)
    a_02 = fem.form(a_02, entity_maps=entity_maps)
    a_20 = fem.form(a_20, entity_maps=entity_maps)
    a_22 = fem.form(a_22, entity_maps=entity_maps)

    L_0 = fem.form(inner(f + u_n / delta_t, v) * dx_c)
    L_1 = fem.form(inner(fem.Constant(msh, 0.0), q) * dx_c)
    L_2 = fem.form(L_2, entity_maps=entity_maps)
    L_3 = fem.form(inner(fem.Constant(
        facet_mesh, PETSc.ScalarType(0.0)), qbar) * dx_f)

    a = [[a_00, a_01, a_02, a_03],
         [a_10, None, None, None],
         [a_20, None, a_22, a_23],
         [a_30, None, a_32, None]]
    L = [L_0, L_1, L_2, L_3]

    return a, L, bcs, bc_funcs


def compute_offsets(V, Q, Vbar):
    u_offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    p_offset = u_offset + \
        Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    ubar_offset = \
        p_offset + Vbar.dofmap.index_map.size_local * \
        Vbar.dofmap.index_map_bs
    return u_offset, p_offset, ubar_offset


def solve(solver_type, k, nu, num_time_steps,
          delta_t, scheme, msh, mt, boundaries,
          boundary_conditions, f, u_i_expr, u_e=None,
          p_e=None):
    facet_mesh, entity_map = create_facet_mesh(msh)

    V, Q, Vbar, Qbar = create_function_spaces(msh, facet_mesh, scheme, k)

    u_n = fem.Function(V)
    u_n.interpolate(u_i_expr)
    ubar_n = fem.Function(Vbar)
    ubar_n.interpolate(u_i_expr)

    a, L, bcs, bc_funcs = create_forms(
        V, Q, Vbar, Qbar, msh, k, delta_t, nu,
        entity_map, solver_type, boundary_conditions,
        boundaries, mt, f, facet_mesh, u_n, ubar_n)

    if solver_type == SolverType.NAVIER_STOKES:
        A = fem.petsc.create_matrix_block(a)
    else:
        A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
        A.assemble()

    if scheme == Scheme.RW:
        u_vis = fem.Function(V)
    else:
        V_vis = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k + 1))
        u_vis = fem.Function(V_vis)
    u_vis.name = "u"
    u_vis.interpolate(u_n)
    p_h = fem.Function(Q)
    p_h.name = "p"
    pbar_h = fem.Function(Qbar)
    pbar_h.name = "pbar"

    u_offset, p_offset, ubar_offset = compute_offsets(V, Q, Vbar)

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

    # Set up files for visualisation
    vis_files = [io.VTXWriter(msh.comm, file_name, [func._cpp_object])
                 for (file_name, func)
                 in [("u.bp", u_vis), ("p.bp", p_h), ("ubar.bp", ubar_n),
                 ("pbar.bp", pbar_h)]]

    t = 0.0
    for vis_file in vis_files:
        vis_file.write(t)
    for n in range(num_time_steps):
        t += delta_t
        par_print(f"t = {t}")

        for bc_func, bc_expr in bc_funcs:
            if isinstance(bc_expr, TimeDependentExpression):
                bc_expr.t = t
                bc_func.interpolate(bc_expr)

        if solver_type == SolverType.NAVIER_STOKES:
            A.zeroEntries()
            fem.petsc.assemble_matrix_block(A, a, bcs=bcs)
            A.assemble()

        with b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

        # Compute solution
        ksp.solve(b, x)

        u_n.x.array[:u_offset] = x.array_r[:u_offset]
        u_n.x.scatter_forward()
        p_h.x.array[:p_offset - u_offset] = x.array_r[u_offset:p_offset]
        p_h.x.scatter_forward()
        ubar_n.x.array[:ubar_offset -
                       p_offset] = x.array_r[p_offset:ubar_offset]
        ubar_n.x.scatter_forward()
        pbar_h.x.array[:(len(x.array_r) - ubar_offset)
                       ] = x.array_r[ubar_offset:]
        pbar_h.x.scatter_forward()

        u_vis.interpolate(u_n)

        for vis_file in vis_files:
            vis_file.write(t)

    for vis_file in vis_files:
        vis_file.close()

    e_div_u = norm_L2(msh.comm, div(u_n))
    e_jump_u = normal_jump_error(msh, u_n)
    par_print(f"e_div_u = {e_div_u}")
    par_print(f"e_jump_u = {e_jump_u}")

    x = ufl.SpatialCoordinate(msh)
    xbar = ufl.SpatialCoordinate(facet_mesh)
    if u_e is not None:
        e_u = norm_L2(msh.comm, u_n - u_e(x))
        e_ubar = norm_L2(msh.comm, ubar_n - u_e(xbar))
        par_print(f"e_u = {e_u}")
        par_print(f"e_ubar = {e_ubar}")

    # par_print(1 / msh.topology.index_map(tdim).size_global**(1 / tdim))

    if p_e is not None:
        p_h_avg = domain_average(msh, p_h)
        p_e_avg = domain_average(msh, p_e(x))
        e_p = norm_L2(msh.comm, (p_h - p_h_avg) - (p_e(x) - p_e_avg))
        pbar_h_avg = domain_average(facet_mesh, pbar_h)
        pbar_e_avg = domain_average(facet_mesh, p_e(xbar))
        e_pbar = norm_L2(msh.comm, (pbar_h - pbar_h_avg) -
                         (p_e(xbar) - pbar_e_avg))

        par_print(f"e_p = {e_p}")
        par_print(f"e_pbar = {e_pbar}")


class Problem:
    def create_mesh(self, h):
        pass

    def u_e(self, x):
        return None

    def p_e(self, x):
        return None

    def boundary_conditions(self):
        pass

    def f(self, msh):
        pass


class GaussianBump(Problem):
    def create_mesh(self, h, cell_type):
        def gaussian(x, a, sigma, mu):
            return a * np.exp(- 1 / 2 * ((x - mu) / sigma)**2)

        comm = MPI.COMM_WORLD
        gdim = 2

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
                for x in np.linspace(0.0, w, num_bottom_points)]
            top_left_point = gmsh.model.geo.addPoint(0, 1, 0, h)
            top_right_point = gmsh.model.geo.addPoint(w, 1, 0, h)

            # Line tags
            lines = []
            lines.append(gmsh.model.geo.addSpline(bottom_points))
            lines.append(gmsh.model.geo.addLine(bottom_points[-1],
                                                top_right_point))
            lines.append(gmsh.model.geo.addLine(top_right_point,
                                                top_left_point))
            lines.append(gmsh.model.geo.addLine(top_left_point,
                                                bottom_points[0]))

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
        msh, _, mt = gmshio.model_to_mesh(
            gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
        gmsh.finalize()

        boundaries = {"left": 4,
                      "right": 2,
                      "bottom": 1,
                      "top": 3}
        return msh, mt, boundaries

    def boundary_conditions(self):
        def inlet(x): return np.vstack(
            (np.ones_like(x[0]),
             np.zeros_like(x[0])))

        def zero(x): return np.vstack(
            (np.zeros_like(x[0]),
             np.zeros_like(x[0])))

        return {"left": (BCType.Dirichlet, inlet),
                "right": (BCType.Neumann, zero),
                "bottom": (BCType.Dirichlet, zero),
                "top": (BCType.Dirichlet, zero)}

    def f(self, msh):
        return fem.Constant(msh, (PETSc.ScalarType(0.0),
                                  PETSc.ScalarType(0.0)))

    def u_i(self):
        return lambda x: np.zeros_like(x[:2])


class Cylinder(Problem):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def create_mesh(self, h, cell_type):
        comm = MPI.COMM_WORLD

        volume_id = {"fluid": 1}

        boundary_id = {"inlet": 2,
                       "outlet": 3,
                       "wall": 4,
                       "obstacle": 5}

        gmsh.initialize()
        if comm.rank == 0:
            gmsh.model.add("model")
            factory = gmsh.model.geo

            if self.d == 2:
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
                factory.addPoint(0.0, height, 0.0, h)
            ]

            thetas = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4,
                      7 * np.pi / 4, 9 * np.pi / 4]
            circle_points = [factory.addPoint(c[0], c[1], 0.0)] + \
                [factory.addPoint(c[0] + r * np.cos(theta),
                                  c[1] + r * np.sin(theta), 0.0)
                 for theta in thetas]

            square_points = [
                factory.addPoint(c[0] + r_s * np.cos(theta),
                                 c[1] + r_s * np.sin(theta), 0.0)
                for theta in thetas]

            rectangle_lines = [
                factory.addLine(rectangle_points[0], rectangle_points[1]),
                factory.addLine(rectangle_points[1], rectangle_points[2]),
                factory.addLine(rectangle_points[2], rectangle_points[3]),
                factory.addLine(rectangle_points[3], rectangle_points[0])
            ]

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

            rectangle_curve = factory.addCurveLoop(rectangle_lines)
            factory.addCurveLoop(circle_lines)
            square_curve = factory.addCurveLoop(square_lines)
            boundary_layer_curves = [
                factory.addCurveLoop(bll) for bll in boundary_layer_lines]

            outer_surface = factory.addPlaneSurface(
                [rectangle_curve, square_curve])
            boundary_layer_surfaces = [
                factory.addPlaneSurface([blc])
                for blc in boundary_layer_curves]

            num_bl_eles = round(0.5 * 1 / h)
            progression_coeff = 1.2
            for i in range(len(boundary_layer_surfaces)):
                gmsh.model.geo.mesh.setTransfiniteCurve(
                    boundary_layer_lines[i][0], num_bl_eles)
                gmsh.model.geo.mesh.setTransfiniteCurve(
                    boundary_layer_lines[i][1], num_bl_eles,
                    coef=progression_coeff)
                gmsh.model.geo.mesh.setTransfiniteCurve(
                    boundary_layer_lines[i][2], num_bl_eles)
                gmsh.model.geo.mesh.setTransfiniteCurve(
                    boundary_layer_lines[i][3], num_bl_eles,
                    coef=progression_coeff)
                gmsh.model.geo.mesh.setTransfiniteSurface(
                    boundary_layer_surfaces[i])

            # FIXME Don't recombine for tets
            if self.d == 3:
                if cell_type == mesh.CellType.tetrahedron:
                    recombine = False
                else:
                    recombine = True
                extrude_surfs = [(2, surf) for surf in [
                    outer_surface] + boundary_layer_surfaces]
                gmsh.model.geo.extrude(
                    extrude_surfs, 0, 0, 0.41, [8], recombine=recombine)

            gmsh.model.geo.synchronize()

            if self.d == 2:
                gmsh.model.addPhysicalGroup(
                    2, [outer_surface] + boundary_layer_surfaces,
                    volume_id["fluid"])

                gmsh.model.addPhysicalGroup(
                    1, [rectangle_lines[0], rectangle_lines[2]],
                    boundary_id["wall"])
                gmsh.model.addPhysicalGroup(
                    1, [rectangle_lines[1]], boundary_id["outlet"])
                gmsh.model.addPhysicalGroup(
                    1, [rectangle_lines[3]], boundary_id["inlet"])
                gmsh.model.addPhysicalGroup(
                    1, circle_lines, boundary_id["obstacle"])
            else:
                # FIXME Mark without hardcoding
                gmsh.model.addPhysicalGroup(
                    3, [1, 2, 3, 4, 5], volume_id["fluid"])

                gmsh.model.addPhysicalGroup(
                    2, [41], boundary_id["inlet"])

                gmsh.model.addPhysicalGroup(
                    2, [33], boundary_id["outlet"])

                gmsh.model.addPhysicalGroup(
                    2, [1, 2, 3, 4, 5, 29, 37, 58, 80, 102, 124, 146],
                    boundary_id["wall"])

                gmsh.model.addPhysicalGroup(
                    2, [75, 97, 119, 141],
                    boundary_id["obstacle"])

            # gmsh.option.setNumber("Mesh.Smoothing", 5)
            if cell_type == mesh.CellType.quadrilateral \
                    or cell_type == mesh.CellType.hexahedron:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 8)
                # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.model.mesh.generate(self.d)
            gmsh.model.mesh.setOrder(order)

        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
        msh, _, mt = gmshio.model_to_mesh(
            gmsh.model, comm, 0, gdim=self.d, partitioner=partitioner)
        gmsh.finalize()

        return msh, mt, boundary_id

    def boundary_conditions(self):
        if self.d == 2:
            def inlet(x): return np.vstack(
                ((1.5 * 4 * x[1] * (0.41 - x[1])) / 0.41**2,
                 np.zeros_like(x[0])))

            def zero(x): return np.vstack(
                (np.zeros_like(x[0]),
                 np.zeros_like(x[0])))
        else:
            H = 0.41

            def inlet(x): return np.vstack(
                (16 * 0.45 * x[1] * x[2] * (H - x[1]) * (H - x[2]) / H**4,
                 np.zeros_like(x[0]),
                 np.zeros_like(x[0])))

            def zero(x): return np.vstack(
                (np.zeros_like(x[0]),
                 np.zeros_like(x[0]),
                 np.zeros_like(x[0])))

        return {"inlet": (BCType.Dirichlet, inlet),
                "outlet": (BCType.Neumann, zero),
                "wall": (BCType.Dirichlet, zero),
                "obstacle": (BCType.Dirichlet, zero)}

    def f(self, msh):
        return fem.Constant(
            msh, [PETSc.ScalarType(0.0) for i in range(self.d)])

    def u_i(self):
        # FIXME Should be tdim
        return lambda x: np.zeros_like(x[:self.d])


class Square(Problem):
    def __init__(self, d=2):
        super().__init__()
        self.d = d

    def create_mesh(self, h, cell_type):
        comm = MPI.COMM_WORLD
        n = round(1 / h)
        if self.d == 2:
            msh = mesh.create_unit_square(
                comm, n, n, cell_type, mesh.GhostMode.none)

            def boundary_marker(x):
                return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | \
                    np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
        else:
            msh = mesh.create_unit_cube(
                comm, n, n, n, cell_type, mesh.GhostMode.none)

            def boundary_marker(x):
                return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | \
                    np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) | \
                    np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)

        fdim = msh.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            msh, fdim, boundary_marker)
        perm = np.argsort(boundary_facets)
        values = np.ones_like(boundary_facets, dtype=np.intc)
        mt = mesh.meshtags(
            msh, fdim, boundary_facets[perm], values[perm])

        boundaries = {"boundary": 1}
        return msh, mt, boundaries

    def u_e(self, x, module=ufl):
        if self.d == 2:
            u = (module.sin(module.pi * x[0]) * module.sin(module.pi * x[1]),
                 module.cos(module.pi * x[0]) * module.cos(module.pi * x[1]))
        else:
            u = (module.sin(module.pi * x[0]) * module.cos(module.pi * x[1])
                 - module.sin(module.pi * x[0]) * module.cos(module.pi * x[2]),
                 module.sin(module.pi * x[1]) * module.cos(module.pi * x[2])
                 - module.sin(module.pi * x[1]) * module.cos(module.pi * x[0]),
                 module.sin(module.pi * x[2]) * module.cos(module.pi * x[0])
                 - module.sin(module.pi * x[2]) * module.cos(module.pi * x[1]))
        if module == ufl:
            return ufl.as_vector(u)
        else:
            assert module == np
            return np.vstack(u)

    def p_e(self, x, module=ufl):
        if self.d == 2:
            return module.sin(module.pi * x[0]) * module.cos(module.pi * x[1])
        else:
            return module.sin(module.pi * x[0]) \
                * module.cos(module.pi * x[1]) * module.sin(module.pi * x[2])
        # return x[0] * (1 - x[0])

    def boundary_conditions(self):
        def u_bc(x): return self.u_e(x, module=np)
        return {"boundary": (BCType.Dirichlet, u_bc)}

    def f(self, msh):
        x = ufl.SpatialCoordinate(msh)
        f = - nu * div(grad(self.u_e(x))) + grad(self.p_e(x))
        if solver_type == SolverType.NAVIER_STOKES:
            f += div(outer(self.u_e(x), self.u_e(x)))
        return f

    def u_i(self):
        return lambda x: np.zeros_like(x[:self.d])


class TaylorGreen(Problem):
    def __init__(self, Re, t_end):
        super().__init__()
        self.Re = Re
        self.t_end = t_end

    def create_mesh(self, h, cell_type):
        comm = MPI.COMM_WORLD
        n = round(1 / h)
        point_0 = (- np.pi / 2, - np.pi / 2)
        point_1 = (np.pi / 2, np.pi / 2)
        msh = mesh.create_rectangle(
            comm, (point_0, point_1), (n, n), cell_type, mesh.GhostMode.none)

        fdim = msh.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            msh, fdim,
            lambda x: np.isclose(x[0], point_0[0]) |
            np.isclose(x[0], point_1[0]) |
            np.isclose(x[1], point_0[1]) |
            np.isclose(x[1], point_1[1]))
        values = np.ones_like(boundary_facets, dtype=np.intc)
        mt = mesh.meshtags(msh, fdim, boundary_facets, values)

        boundaries = {"boundary": 1}
        return msh, mt, boundaries

    def u_expr(self, x, t, module):
        return (- module.cos(x[0]) * module.sin(x[1]) *
                module.exp(- 2 * t / self.Re),
                module.sin(x[0]) * module.cos(x[1]) *
                module.exp(- 2 * t / self.Re))

    def u_e(self, x, module=ufl):
        return ufl.as_vector(self.u_expr(x, self.t_end, ufl))

    def p_e(self, x):
        return - 1 / 4 * (ufl.cos(2 * x[0]) + ufl.cos(2 * x[1])) * ufl.exp(
            - 4 * self.t_end / self.Re)

    def boundary_conditions(self):
        u_bc = TimeDependentExpression(
            lambda x, t: np.vstack(self.u_expr(x, t, np)))
        return {"boundary": (BCType.Dirichlet, u_bc)}

    def u_i(self):
        return lambda x: self.u_expr(x, t=0, module=np)

    def f(self, msh):
        return ufl.as_vector((0.0, 0.0))


# TODO Remove duplicate code
class Kovasznay(Problem):
    def create_mesh(self, h, cell_type):
        comm = MPI.COMM_WORLD
        n = round(1 / h)

        point_0 = (0.0, -0.5)
        point_1 = (1, 1.5)
        msh = mesh.create_rectangle(
            comm, (point_0, point_1), (n, 2 * n),
            cell_type, mesh.GhostMode.none)

        fdim = msh.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            msh, fdim,
            lambda x: np.isclose(x[0], point_0[0]) |
            np.isclose(x[0], point_1[0]) |
            np.isclose(x[1], point_0[1]) |
            np.isclose(x[1], point_1[1]))
        values = np.ones_like(boundary_facets, dtype=np.intc)
        mt = mesh.meshtags(msh, fdim, boundary_facets, values)

        boundaries = {"boundary": 1}
        return msh, mt, boundaries

    def u_e(self, x, module=ufl):
        R_e = 1 / nu
        u_x = 1 - module.exp(
            (R_e / 2 - module.sqrt(R_e**2 / 4 + 4 * module.pi**2)) * x[0]) * \
            module.cos(2 * module.pi * x[1])
        u_y = (R_e / 2 - module.sqrt(R_e**2 / 4 + 4 * module.pi**2)) / \
            (2 * module.pi) * module.exp(
            (R_e / 2 - module.sqrt(R_e**2 / 4 + 4 * module.pi**2)) * x[0]) * \
            module.sin(2 * module.pi * x[1])
        if module == ufl:
            return ufl.as_vector((u_x, u_y))
        else:
            assert module == np
            return np.vstack((u_x, u_y))

    def p_e(self, x, module=ufl):
        R_e = 1 / nu
        return (1 / 2) * (1 - module.exp(
            2 * (R_e / 2 - module.sqrt(R_e**2 / 4 + 4 * module.pi**2)) * x[0]))

    def boundary_conditions(self):
        def u_bc(x): return self.u_e(x, module=np)
        return {"boundary": (BCType.Dirichlet, u_bc)}

    def f(self, msh):
        return fem.Constant(msh, (PETSc.ScalarType(0.0),
                                  PETSc.ScalarType(0.0)))

    def u_i(self):
        return lambda x: np.zeros_like(x[:2])


class Wannier(Problem):
    def __init__(self, r_0=0.7, r_1=1.0, e=-0.15, v_0=1.0, v_1=0):
        super().__init__()
        self.r_0 = r_0
        self.r_1 = r_1
        self.e = e
        self.v_0 = v_0
        self.v_1 = v_1

    def create_mesh(self, h, cell_type, order=1):
        comm = MPI.COMM_WORLD
        gdim = 2

        volume_id = {"fluid": 1}
        boundary_id = {"wall_0": 2,
                       "wall_1": 3}

        gmsh.initialize()
        if comm.rank == 0:
            gmsh.model.add("model")
            factory = gmsh.model.occ

            circle_0 = factory.addDisk(0.0, self.e, 0.0, self.r_0, self.r_0)
            circle_1 = factory.addDisk(0.0, 0.0, 0.0, self.r_1, self.r_1)

            ov, ovv = factory.cut([(2, circle_1)], [(2, circle_0)])

            gmsh.model.occ.synchronize()

            boundary_dim_tags = gmsh.model.getBoundary([ov[0]], oriented=False)

            gmsh.model.addPhysicalGroup(2, [ov[0][1]], volume_id["fluid"])
            gmsh.model.addPhysicalGroup(
                1, [boundary_dim_tags[1][1]], boundary_id["wall_0"])
            gmsh.model.addPhysicalGroup(
                1, [boundary_dim_tags[0][1]], boundary_id["wall_1"])

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

            boundary_dim_tags = gmsh.model.getBoundary([ov[0]], oriented=False)
            gmsh.option.setNumber("Mesh.Smoothing", 5)
            if cell_type == mesh.CellType.quadrilateral:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 8)
                gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(order)

        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
        msh, _, mt = gmshio.model_to_mesh(
            gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
        gmsh.finalize()

        return msh, mt, boundary_id

    def boundary_conditions(self):
        def u_D_0(x):
            r = np.vstack((x[0], x[1] - self.e))
            r_hat = r / self.r_0
            t_hat = np.vstack((
                - (r_hat[1]),
                r_hat[0]
            ))
            return self.v_0 * t_hat

        def u_D_1(x):
            r_hat = x / self.r_1
            t_hat = np.vstack((-r_hat[1], r_hat[0]))
            return self.v_1 * t_hat

        return {"wall_0": (BCType.Dirichlet, u_D_0),
                "wall_1": (BCType.Dirichlet, u_D_1)}

    def f(self, msh):
        return fem.Constant(msh, (PETSc.ScalarType(0.0),
                                  PETSc.ScalarType(0.0)))

    def u_e(self, x, module=ufl):
        r_0 = self.r_0
        r_1 = self.r_1
        e = - self.e
        v_0 = self.v_0
        v_1 = self.v_1

        d_0 = (r_1 * r_1 - r_0 * r_0) / (2 * e) - e / 2
        d_1 = d_0 + e
        s = module.sqrt((r_1 - r_0 - e) * (r_1 - r_0 + e) *
                        (r_1 + r_0 + e) * (r_1 + r_0 - e)) / (2 * e)
        l_0 = module.ln((d_0 + s) / (d_0 - s))
        l_1 = module.ln((d_1 + s) / (d_1 - s))
        den = (r_1 * r_1 + r_0 * r_0) * (l_0 - l_1) - 4 * s * e
        curlb = 2 * (d_1 * d_1 - d_0 * d_0) * (r_0 * v_0 + r_1 * v_1) \
            / ((r_1 * r_1 + r_0 * r_0) * den) + r_0 * r_0 * r_1 * r_1 \
            * (v_0 / r_0 - v_1 / r_1) / (s * (r_0 * r_0 + r_1 * r_1)
                                         * (d_1 - d_0))
        A = - 0.5 * (d_0 * d_1 - s * s) * curlb
        B = (d_0 + s) * (d_1 + s) * curlb
        C = (d_0 - s) * (d_1 - s) * curlb
        D = (d_0 * l_1 - d_1 * l_0) * (r_0 * v_0 + r_1 * v_1) / den - 2 * s \
            * ((r_1 * r_1 - r_0 * r_0) / (r_1 * r_1 + r_0 * r_0)) \
            * (r_0 * v_0 + r_1 * v_1) / den - r_0 * r_0 * r_1 * r_1 \
            * (v_0 / r_0 - v_1 / r_1) / ((r_0 * r_0 + r_1 * r_1) * e)
        E = 0.5 * (l_0 - l_1) * (r_0 * v_0 + r_1 * v_1) / den
        F = e * (r_0 * v_0 + r_1 * v_1) / den

        y_offset = x[1] + d_1
        spy = s + y_offset
        smy = s - y_offset
        zp = x[0] * x[0] + spy * spy
        zm = x[0] * x[0] + smy * smy
        lz = module.ln(zp / zm)
        zr = 2 * (spy / zp + smy / zm)

        u_x = - A * zr - B * ((s + 2 * y_offset) * zp - 2 * spy * spy
                              * y_offset) / (zp * zp) - C * (
            (s - 2 * y_offset)
            * zm + 2 * smy
            * smy * y_offset) \
            / (zm * zm) - D - E * 2 * y_offset - F * (lz + y_offset * zr)
        u_y = - A * 8 * s * x[0] * y_offset / (zp * zm) \
            - B * 2 * x[0] * y_offset * spy / (zp * zp) - C * 2 \
            * x[0] * y_offset * smy / (zm * zm) + E * 2 * x[0] - F * 8 * s \
            * x[0] * y_offset * y_offset / (zp * zm)

        if module == ufl:
            return ufl.as_vector((u_x, u_y))
        else:
            assert module == np
            return np.vstack((u_x, u_y))

    def u_i(self):
        return lambda x: np.zeros_like(x[:2])


if __name__ == "__main__":
    # Simulation parameters
    solver_type = SolverType.NAVIER_STOKES
    h = 1 / 16
    k = 3
    cell_type = mesh.CellType.quadrilateral
    nu = 1.0e-3
    num_time_steps = 32
    t_end = 1e4
    delta_t = t_end / num_time_steps
    scheme = Scheme.DRW

    comm = MPI.COMM_WORLD
    problem = Square()
    msh, mt, boundaries = problem.create_mesh(h, cell_type)
    boundary_conditions = problem.boundary_conditions()
    u_i_expr = problem.u_i()
    f = problem.f(msh)

    solve(solver_type, k, nu, num_time_steps,
          delta_t, scheme, msh, mt, boundaries,
          boundary_conditions, f, u_i_expr, problem.u_e,
          problem.p_e)
