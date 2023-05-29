import hdg_navier_stokes
from hdg_navier_stokes import SolverType, Scheme, TimeDependentExpression
from mpi4py import MPI
import gmsh
import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.io import gmshio
from hdg_navier_stokes import BCType
from petsc4py import PETSc
from utils import norm_L2, normal_jump_error, domain_average
import ufl
from ufl import (div, TrialFunction, TestFunction, inner, curl, cross,
                 as_vector, grad, outer, dot)
import sys
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.cpp.fem import compute_integration_domains


def par_print(string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


class GaussianBump(hdg_navier_stokes.Problem):
    def __init__(self, d) -> None:
        super().__init__()
        self.d = d

    def create_mesh(self, h, cell_type):
        comm = MPI.COMM_WORLD

        boundaries = {"inlet": 1,
                      "outlet": 2,
                      "walls": 3}

        gmsh.initialize()
        if comm.rank == 0:
            # TODO Pass options
            gmsh.model.add("channel")
            order = 1

            length = 2
            height = 1
            points = [gmsh.model.geo.addPoint(0, 0, 0, h),
                      gmsh.model.geo.addPoint(length, 0, 0, h),
                      gmsh.model.geo.addPoint(length, height, 0, h),
                      gmsh.model.geo.addPoint(0, height, 0, h)]

            # Line tags
            lines = [gmsh.model.geo.addLine(points[0], points[1]),
                     gmsh.model.geo.addLine(points[1], points[2]),
                     gmsh.model.geo.addLine(points[2], points[3]),
                     gmsh.model.geo.addLine(points[3], points[0])]

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

            if self.d == 3:
                if cell_type == mesh.CellType.tetrahedron:
                    recombine = False
                else:
                    recombine = True
                extrude_surfs = [(2, 1)]
                gmsh.model.geo.extrude(
                    extrude_surfs, 0, 0, 1.0, [round(height / h)], recombine=recombine)

            gmsh.model.geo.synchronize()

            if self.d == 2:
                gmsh.model.addPhysicalGroup(2, [1], 1)

                gmsh.model.addPhysicalGroup(
                    1, [lines[0], lines[2]], boundaries["walls"])
                gmsh.model.addPhysicalGroup(
                    1, [lines[1]], boundaries["outlet"])
                gmsh.model.addPhysicalGroup(1, [lines[3]], boundaries["inlet"])
            else:
                gmsh.model.addPhysicalGroup(3, [1], 1)
                gmsh.model.addPhysicalGroup(
                    2, [1, 13, 21, 26], boundaries["walls"])
                gmsh.model.addPhysicalGroup(2, [25], boundaries["inlet"])
                gmsh.model.addPhysicalGroup(2, [17], boundaries["outlet"])

            gmsh.option.setNumber("Mesh.Smoothing", 5)
            if cell_type == mesh.CellType.quadrilateral \
                    or cell_type == mesh.CellType.hexahedron:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 8)
                # TODO Check what this is doing, it may be making things worse
                gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.model.mesh.generate(self.d)
            gmsh.model.mesh.setOrder(order)

            # gmsh.write("msh.msh")
        # exit()

        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
        msh, _, ft = gmshio.model_to_mesh(
            gmsh.model, comm, 0, gdim=self.d, partitioner=partitioner)
        gmsh.finalize()

        return msh, ft, boundaries

    def boundary_conditions(self):
        if self.d == 2:
            def inlet(x): return np.vstack(
                (6 * x[1] * (1 - x[1]),
                 np.zeros_like(x[0])))

            def zero(x): return np.vstack(
                (np.zeros_like(x[0]),
                 np.zeros_like(x[0])))
        else:
            def inlet(x): return np.vstack(
                (36 * x[1] * (1 - x[1]) * x[2] * (1 - x[2]),
                 np.zeros_like(x[0]),
                 np.zeros_like(x[0])))

            def zero(x): return np.vstack(
                (np.zeros_like(x[0]),
                 np.zeros_like(x[0]),
                 np.zeros_like(x[0])))

        return {"inlet": (BCType.Dirichlet, inlet),
                "outlet": (BCType.Neumann, zero),
                "walls": (BCType.Dirichlet, zero)}

    def f(self, msh):
        return fem.Constant(msh, [PETSc.ScalarType(0.0) for i in range(self.d)])

    def u_i(self):
        return lambda x: np.zeros_like(x[:d])


def solve(solver_type, k, nu, num_time_steps,
          delta_t, scheme, msh, mt, boundaries,
          boundary_conditions, f, u_i_expr, u_e=None,
          p_e=None):
    facet_mesh, entity_map = hdg_navier_stokes.create_facet_mesh(msh)

    V, Q, Vbar, Qbar = hdg_navier_stokes.create_function_spaces(
        msh, facet_mesh, scheme, k)

    sigma = fem.Constant(msh, 2.0)
    mu = fem.Constant(msh, 0.5)

    # H(curl; Ω)-conforming space for magnetic vector potential and electric
    # field
    X = fem.FunctionSpace(msh, ("Nedelec 1st kind H(curl)", k + 1))

    u_n = fem.Function(V)
    u_n.interpolate(u_i_expr)
    ubar_n = fem.Function(Vbar)
    ubar_n.interpolate(u_i_expr)

    # Trial and test functions for the magnetic vector potential
    A, phi = TrialFunction(X), TestFunction(X)
    A_h = fem.Function(X)
    A_n = fem.Function(X)

    B_0 = as_vector((0, 1, 0))

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
    gamma = 32.0 * k**2 / h  # TODO Should be larger in 3D

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

    # Using linearised version (3.91) https://academic.oup.com/book/5953/chapter/149296535?login=true
    a_44 = fem.form(inner(sigma * A / delta_t, phi) * dx_c
                    + inner(1 / mu * curl(A), curl(phi)) * dx_c)
    a_40 = fem.form(inner(sigma * cross(B_0, u), phi) * dx_c
                    + inner(sigma * cross(curl(A_n), u), phi) * dx_c)
    a_04 = fem.form(
        inner(sigma * A / delta_t, cross(curl(A_n), v)) * dx_c
        - inner(sigma * cross(u_n, curl(A)), cross(curl(A_n), v)) * dx_c
        + inner(sigma * A / delta_t, cross(B_0, v)) * dx_c)

    L_4 = fem.form(inner(sigma * A_n / delta_t, phi) * dx_c)

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

    L_0 = fem.form(inner(f + u_n / delta_t, v) * dx_c
                   + inner(sigma * A_n / delta_t, cross(curl(A_n), v)) * dx_c
                   + inner(sigma * A_n / delta_t, cross(B_0, v)) * dx_c
                   + inner(sigma * cross(u_n, B_0), cross(B_0, v)) * dx_c)

    L_1 = fem.form(inner(fem.Constant(msh, 0.0), q) * dx_c)
    L_2 = fem.form(L_2, entity_maps=entity_maps)
    L_3 = fem.form(inner(fem.Constant(
        facet_mesh, PETSc.ScalarType(0.0)), qbar) * dx_f)

    a = [[a_00, a_01, a_02, a_03, a_04],
         [a_10, None, None, None, None],
         [a_20, None, a_22, a_23, None],
         [a_30, None, a_32, None, None],
         [a_40, None, None, None, a_44]]
    L = [L_0, L_1, L_2, L_3, L_4]

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

    B_h = B_0 + curl(A_h)
    B_expr = fem.Expression(B_h, V_vis.element.interpolation_points())
    B_vis = fem.Function(V_vis)
    B_vis.interpolate(B_expr)

    u_offset, p_offset, ubar_offset = hdg_navier_stokes.compute_offsets(
        V, Q, Vbar)
    pbar_offset = ubar_offset + Qbar.dofmap.index_map.size_local * \
        Qbar.dofmap.index_map_bs

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
                 ("pbar.bp", pbar_h), ("B.bp", B_vis)]]

    t = 0.0
    for vis_file in vis_files:
        vis_file.write(t)
    for n in range(num_time_steps):
        t += delta_t.value
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
        pbar_h.x.array[:pbar_offset -
                       ubar_offset] = x.array_r[ubar_offset:pbar_offset]
        pbar_h.x.scatter_forward()
        A_h.x.array[:(len(x.array_r) - pbar_offset)
                    ] = x.array_r[pbar_offset:]
        A_h.x.scatter_forward()

        B_vis.interpolate(B_expr)

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


if __name__ == "__main__":
    # Simulation parameters
    solver_type = SolverType.NAVIER_STOKES
    h = 1 / 8
    k = 1
    cell_type = mesh.CellType.hexahedron
    nu = 1.0e-3
    num_time_steps = 10
    t_end = 10
    delta_t = t_end / num_time_steps
    scheme = Scheme.DRW

    if cell_type == mesh.CellType.tetrahedron or \
            cell_type == mesh.CellType.hexahedron:
        d = 3
    else:
        d = 2

    comm = MPI.COMM_WORLD
    problem = GaussianBump(d)
    msh, ft, boundaries = problem.create_mesh(h, cell_type)

    # with io.XDMFFile(msh.comm, "msh.xdmf", "w") as file:
    #     file.write_mesh(msh)
    #     # file.write_meshtags(ct)
    #     file.write_meshtags(ft)

    boundary_conditions = problem.boundary_conditions()
    u_i_expr = problem.u_i()
    f = problem.f(msh)

    solve(solver_type, k, nu, num_time_steps,
          delta_t, scheme, msh, ft, boundaries,
          boundary_conditions, f, u_i_expr, problem.u_e,
          problem.p_e)
