import hdg_navier_stokes
from hdg_navier_stokes import SolverType, Scheme, TimeDependentExpression
from mpi4py import MPI
import gmsh
import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.io import gmshio
from hdg_navier_stokes import BCType
from petsc4py import PETSc
from utils import norm_L2, normal_jump_error, domain_average, convert_facet_tags
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


class Channel(hdg_navier_stokes.Problem):
    def create_mesh(self, n_x, n_y, n_z, n_s_y, cell_type):
        comm = MPI.COMM_WORLD

        volumes = {"solid": 1,
                   "fluid": 2}

        boundaries = {"solid_x_walls": 1,
                      "solid_y_walls": 2,
                      "solid_z_walls": 3,
                      "fluid_y_walls": 4,
                      "fluid_z_walls": 5,
                      "inlet": 6,
                      "outlet": 7}

        gmsh.initialize()
        if comm.rank == 0:
            # TODO Pass options
            gmsh.model.add("channel")
            order = 1

            # Fluid domain x, y, and z lengths
            L_x = 4
            L_y = 1
            L_z = 1
            wall_thickness = 0.1
            domain_points = [gmsh.model.geo.addPoint(0, - wall_thickness, 0),
                             gmsh.model.geo.addPoint(L_x, - wall_thickness, 0),
                             gmsh.model.geo.addPoint(
                                 L_x, L_y + wall_thickness, 0),
                             gmsh.model.geo.addPoint(0, L_y + wall_thickness, 0)]
            fluid_points = [gmsh.model.geo.addPoint(0, 0, 0),
                            gmsh.model.geo.addPoint(L_x, 0, 0),
                            gmsh.model.geo.addPoint(L_x, L_y, 0),
                            gmsh.model.geo.addPoint(0, L_y, 0)]

            wall_0_lines = [gmsh.model.geo.addLine(domain_points[0], domain_points[1]),
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
                           gmsh.model.geo.addLine(fluid_points[3], fluid_points[0])]

            wall_1_lines = [- fluid_lines[2],
                            gmsh.model.geo.addLine(
                                fluid_points[2], domain_points[2]),
                            gmsh.model.geo.addLine(
                                domain_points[2], domain_points[3]),
                            gmsh.model.geo.addLine(domain_points[3], fluid_points[3])]

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

        return msh, ct, ft, volumes, boundaries

    def boundary_conditions(self):
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
        # # Homogeneous Dirichlet (conducting) on y walls,
        # # homogeneous Neumann (insulating) on z walls
        # A_bcs = {"y_walls": (BCType.Dirichlet, zero)}

        # Insulating walls
        A_bcs = {}

        return {"u": u_bcs, "A": A_bcs}

    def f(self, msh):
        return fem.Constant(msh, [PETSc.ScalarType(0.0) for i in range(3)])

    def u_i(self):
        return lambda x: np.zeros_like(x[:3])


def solve(solver_type, k, nu, num_time_steps,
          delta_t, scheme, msh, ct, ft, volumes, boundaries,
          boundary_conditions, f, u_i_expr, u_e=None,
          p_e=None):

    fluid_sm, fluid_sm_ent_map = mesh.create_submesh(
        msh, msh.topology.dim, ct.find(volumes["fluid"]))[:2]

    facet_mesh, entity_map = hdg_navier_stokes.create_facet_mesh(fluid_sm)
    V, Q, Vbar, Qbar = hdg_navier_stokes.create_function_spaces(
        fluid_sm, facet_mesh, scheme, k)

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
    num_cell_facets = cell_num_entities(fluid_sm.topology.cell_type, fdim)
    for cell in range(fluid_sm.topology.index_map(tdim).size_local):
        for local_facet in range(num_cell_facets):
            all_facets.extend([cell, local_facet])

    ft_f = convert_facet_tags(msh, fluid_sm, fluid_sm_ent_map, ft)

    with io.XDMFFile(msh.comm, "sm.xdmf", "w") as file:
        file.write_mesh(fluid_sm)
        file.write_meshtags(ft_f)
    facet_integration_entities = [(all_facets_tag, all_facets)]
    facet_integration_entities += compute_integration_domains(
        fem.IntegralType.exterior_facet, ft_f._cpp_object)
    dx_c = ufl.Measure("dx", domain=fluid_sm)
    # FIXME Figure out why this is being estimated wrong for DRW
    # NOTE k**2 works on affine meshes
    quad_deg = (k + 1)**2
    ds_c = ufl.Measure(
        "ds", subdomain_data=facet_integration_entities, domain=fluid_sm,
        metadata={"quadrature_degree": quad_deg})
    dx_f = ufl.Measure("dx", domain=facet_mesh)

    inv_entity_map = np.full_like(entity_map, -1)
    for i, facet in enumerate(entity_map):
        inv_entity_map[facet] = i

    entity_maps = {facet_mesh: inv_entity_map,
                   msh: fluid_sm_ent_map}

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    ubar = ufl.TrialFunction(Vbar)
    vbar = ufl.TestFunction(Vbar)
    pbar = ufl.TrialFunction(Qbar)
    qbar = ufl.TestFunction(Qbar)

    h = ufl.CellDiameter(fluid_sm)  # TODO Fix for high order geom!
    n = ufl.FacetNormal(fluid_sm)
    gamma = 64.0 * k**2 / h  # TODO Should be larger in 3D

    lmbda = ufl.conditional(ufl.lt(dot(u_n, n), 0), 1, 0)
    delta_t = fem.Constant(msh, PETSc.ScalarType(delta_t))
    nu = fem.Constant(fluid_sm, PETSc.ScalarType(nu))

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
    # a_44 = fem.form(inner(sigma * A / delta_t, phi) * dx_c
    #                 + inner(1 / mu * curl(A), curl(phi)) * dx_c)
    # a_40 = fem.form(inner(sigma * cross(B_0, u), phi) * dx_c
    #                 + inner(sigma * cross(curl(A_n), u), phi) * dx_c)
    # a_04 = fem.form(
    #     inner(sigma * A / delta_t, cross(curl(A_n), v)) * dx_c
    #     - inner(sigma * cross(u_n, curl(A)), cross(curl(A_n), v)) * dx_c
    #     + inner(sigma * A / delta_t, cross(B_0, v)) * dx_c)

    # L_4 = fem.form(inner(sigma * A_n / delta_t, phi) * dx_c)

    L_2 = inner(fem.Constant(fluid_sm, [PETSc.ScalarType(0.0)
                                   for i in range(tdim)]),
                vbar) * ds_c(all_facets_tag)

    # NOTE: Don't set pressure BC to avoid affecting conservation properties.
    # MUMPS seems to cope with the small nullspace
    bcs = []
    bc_funcs = []
    for name, bc in boundary_conditions["u"].items():
        id = boundaries[name]
        bc_type, bc_expr = bc
        bc_func = fem.Function(Vbar)
        bc_func.interpolate(bc_expr)
        bc_funcs.append((bc_func, bc_expr))
        if bc_type == BCType.Dirichlet:
            facets = inv_entity_map[ft_f.indices[ft_f.values == id]]
            dofs = fem.locate_dofs_topological(Vbar, fdim, facets)
            bcs.append(fem.dirichletbc(bc_func, dofs))
        else:
            assert bc_type == BCType.Neumann
            L_2 += inner(bc_func, vbar) * ds_c(id)
            if solver_type == SolverType.NAVIER_STOKES:
                a_22 += - inner((1 - lmbda) * dot(ubar_n, n) *
                                ubar, vbar) * ds_c(id)

    # for name, bc in boundary_conditions["A"].items():
    #     id = boundaries[name]
    #     bc_type, bc_expr = bc
    #     assert bc_type == BCType.Dirichlet
    #     bc_func = fem.Function(X)
    #     bc_func.interpolate(bc_expr)
    #     facets = ft.find(id)
    #     dofs = fem.locate_dofs_topological(X, fdim, facets)
    #     bcs.append(fem.dirichletbc(bc_func, dofs))

    a_00 = fem.form(a_00)
    a_02 = fem.form(a_02, entity_maps=entity_maps)
    a_20 = fem.form(a_20, entity_maps=entity_maps)
    a_22 = fem.form(a_22, entity_maps=entity_maps)

    # L_0 = fem.form(inner(f + u_n / delta_t, v) * dx_c
    #                + inner(sigma * A_n / delta_t, cross(curl(A_n), v)) * dx_c
    #                + inner(sigma * A_n / delta_t, cross(B_0, v)) * dx_c
    #                + inner(sigma * cross(u_n, B_0), cross(B_0, v)) * dx_c)
    L_0 = fem.form(inner(f + u_n / delta_t, v) * dx_c)

    L_1 = fem.form(inner(fem.Constant(msh, 0.0), q) * dx_c)
    L_2 = fem.form(L_2, entity_maps=entity_maps)
    L_3 = fem.form(inner(fem.Constant(
        facet_mesh, PETSc.ScalarType(0.0)), qbar) * dx_f)

    # a = [[a_00, a_01, a_02, a_03, a_04],
    #      [a_10, None, None, None, None],
    #      [a_20, None, a_22, a_23, None],
    #      [a_30, None, a_32, None, None],
    #      [a_40, None, None, None, a_44]]
    # L = [L_0, L_1, L_2, L_3, L_4]

    a = [[a_00, a_01, a_02, a_03],
         [a_10, None, None, None],
         [a_20, None, a_22, a_23],
         [a_30, None, a_32, None]]
    L = [L_0, L_1, L_2, L_3]


    if solver_type == SolverType.NAVIER_STOKES:
        A = fem.petsc.create_matrix_block(a)
    else:
        A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
        A.assemble()

    if scheme == Scheme.RW:
        u_vis = fem.Function(V)
    else:
        V_vis = fem.VectorFunctionSpace(fluid_sm, ("Discontinuous Lagrange", k + 1))
        u_vis = fem.Function(V_vis)
    u_vis.name = "u"
    u_vis.interpolate(u_n)
    p_h = fem.Function(Q)
    p_h.name = "p"
    pbar_h = fem.Function(Qbar)
    pbar_h.name = "pbar"

    # B_h = B_0 + curl(A_h)
    # B_expr = fem.Expression(B_h, V_vis.element.interpolation_points())
    # B_vis = fem.Function(V_vis)
    # B_vis.interpolate(B_expr)

    # J_h = - sigma * ((A_h - A_n) / delta_t + cross(curl(A_h), u_n))
    # J_expr = fem.Expression(J_h, V_vis.element.interpolation_points())
    # J_vis = fem.Function(V_vis)
    # J_vis.interpolate(J_expr)

    u_offset, p_offset, ubar_offset = hdg_navier_stokes.compute_offsets(
        V, Q, Vbar)
    # pbar_offset = ubar_offset + Qbar.dofmap.index_map.size_local * \
    #     Qbar.dofmap.index_map_bs

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
    # vis_files = [io.VTXWriter(msh.comm, file_name, [func._cpp_object])
    #              for (file_name, func)
    #              in [("u.bp", u_vis), ("p.bp", p_h), ("ubar.bp", ubar_n),
    #              ("pbar.bp", pbar_h), ("B.bp", B_vis), ("J.bp", J_vis)]]
    vis_files = [io.VTXWriter(msh.comm, file_name, [func._cpp_object])
                 for (file_name, func)
                 in [("u.bp", u_vis), ("p.bp", p_h), ("ubar.bp", ubar_n),
                 ("pbar.bp", pbar_h)]]

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

        # u_n.x.array[:u_offset] = x.array_r[:u_offset]
        # u_n.x.scatter_forward()
        # p_h.x.array[:p_offset - u_offset] = x.array_r[u_offset:p_offset]
        # p_h.x.scatter_forward()
        # ubar_n.x.array[:ubar_offset -
        #                p_offset] = x.array_r[p_offset:ubar_offset]
        # ubar_n.x.scatter_forward()
        # pbar_h.x.array[:pbar_offset -
        #                ubar_offset] = x.array_r[ubar_offset:pbar_offset]
        # pbar_h.x.scatter_forward()
        # A_h.x.array[:(len(x.array_r) - pbar_offset)
        #             ] = x.array_r[pbar_offset:]
        # A_h.x.scatter_forward()

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

        # B_vis.interpolate(B_expr)
        # J_vis.interpolate(J_expr)

        u_vis.interpolate(u_n)

        for vis_file in vis_files:
            vis_file.write(t)

    for vis_file in vis_files:
        vis_file.close()

    e_div_u = norm_L2(msh.comm, div(u_n))
    e_jump_u = normal_jump_error(fluid_sm, u_n)
    par_print(f"e_div_u = {e_div_u}")
    par_print(f"e_jump_u = {e_jump_u}")


if __name__ == "__main__":
    # Simulation parameters
    solver_type = SolverType.NAVIER_STOKES
    n_x = 12
    n_y = 6
    n_z = 6
    n_s_y = 2

    k = 1
    cell_type = mesh.CellType.hexahedron
    nu = 1.0e-3
    num_time_steps = 32
    t_end = 10
    delta_t = t_end / num_time_steps
    scheme = Scheme.DRW

    comm = MPI.COMM_WORLD
    problem = Channel()
    msh, ct, ft, volumes, boundaries = problem.create_mesh(
        n_x, n_y, n_z, n_s_y, cell_type)

    # with io.XDMFFile(msh.comm, "msh.xdmf", "w") as file:
    #     file.write_mesh(msh)
    #     file.write_meshtags(ct)
    #     file.write_meshtags(ft)
    # exit()

    boundary_conditions = problem.boundary_conditions()
    u_i_expr = problem.u_i()
    f = problem.f(msh)

    solve(solver_type, k, nu, num_time_steps,
          delta_t, scheme, msh, ct, ft, volumes, boundaries,
          boundary_conditions, f, u_i_expr, problem.u_e,
          problem.p_e)
