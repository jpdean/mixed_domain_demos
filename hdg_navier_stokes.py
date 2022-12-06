from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div, outer
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.cpp.fem import compute_integration_domains
from utils import norm_L2, domain_average, normal_jump_error
from enum import Enum
import gmsh
from dolfinx.io import gmshio
import sys


class SolverType(Enum):
    STOKES = 1
    NAVIER_STOKES = 2


class Scheme(Enum):
    RW = 1
    DRW = 2


def par_print(string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def solve(solver_type, k, nu, num_time_steps,
          delta_t, scheme, msh, mt, boundaries,
          boundary_conditions, f, u_e=None,
          p_e=None):
    tdim = msh.topology.dim
    fdim = tdim - 1

    num_cell_facets = cell_num_entities(msh.topology.cell_type, fdim)
    msh.topology.create_entities(fdim)
    facet_imap = msh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    facets = np.arange(num_facets, dtype=np.int32)

    # NOTE Despite all facets being present in the submesh, the entity
    # map isn't necessarily the identity in parallel
    facet_mesh, entity_map = mesh.create_submesh(msh, fdim, facets)[0:2]

    if scheme == Scheme.RW:
        V = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k))
        Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k - 1))
    else:
        V = fem.FunctionSpace(msh, ("Discontinuous Raviart-Thomas", k + 1))
        Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
    Vbar = fem.VectorFunctionSpace(
        facet_mesh, ("Discontinuous Lagrange", k))
    Qbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

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
    gamma = 6.0 * k**2 / h

    facet_integration_entities = compute_integration_domains(
        fem.IntegralType.exterior_facet, mt)

    all_facets = 0
    facet_integration_entities[all_facets] = []
    for cell in range(msh.topology.index_map(tdim).size_local):
        for local_facet in range(num_cell_facets):
            facet_integration_entities[all_facets].extend([cell, local_facet])

    dx_c = ufl.Measure("dx", domain=msh)
    ds_c = ufl.Measure(
        "ds", subdomain_data=facet_integration_entities, domain=msh)
    dx_f = ufl.Measure("dx", domain=facet_mesh)

    inv_entity_map = np.full_like(entity_map, -1)
    for i, facet in enumerate(entity_map):
        inv_entity_map[facet] = i
    entity_maps = {facet_mesh: inv_entity_map}

    u_n = fem.Function(V)
    lmbda = ufl.conditional(ufl.lt(dot(u_n, n), 0), 1, 0)
    delta_t = fem.Constant(msh, PETSc.ScalarType(delta_t))
    nu = fem.Constant(msh, PETSc.ScalarType(nu))

    # TODO Double check convective terms
    a_00 = inner(u / delta_t, v) * dx_c + \
        nu * (inner(grad(u), grad(v)) * dx_c +
              gamma * inner(u, v) * ds_c(all_facets)
              - (inner(u, dot(grad(v), n))
                 + inner(v, dot(grad(u), n))) * ds_c(all_facets))
    a_01 = fem.form(- inner(p, div(v)) * dx_c)
    a_02 = nu * (inner(ubar, dot(grad(v), n)) * ds_c(all_facets)
                 - gamma * inner(ubar, v) * ds_c(all_facets))
    a_03 = fem.form(inner(dot(v, n), pbar) *
                    ds_c(all_facets), entity_maps=entity_maps)
    a_10 = fem.form(- inner(q, div(u)) * dx_c)
    a_20 = nu * (inner(vbar, dot(grad(u), n)) * ds_c(all_facets)
                 - gamma * inner(vbar, u) * ds_c(all_facets))
    a_30 = fem.form(inner(dot(u, n), qbar) *
                    ds_c(all_facets), entity_maps=entity_maps)
    a_22 = nu * gamma * inner(ubar, vbar) * ds_c(all_facets)

    if solver_type == SolverType.NAVIER_STOKES:
        a_00 += inner(outer(u, u_n) - outer(u, lmbda * u_n),
                      outer(v, n)) * ds_c(all_facets) - \
            inner(outer(u, u_n), grad(v)) * dx_c
        a_02 += inner(outer(ubar, lmbda * u_n), outer(v, n)) * ds_c(all_facets)
        a_20 += inner(outer(u, u_n) - outer(u, lmbda * u_n),
                      outer(vbar, n)) * ds_c(all_facets)
        a_22 += inner(outer(ubar, lmbda * u_n),
                      outer(vbar, n)) * ds_c(all_facets)

    a_00 = fem.form(a_00)
    a_02 = fem.form(a_02, entity_maps=entity_maps)
    a_20 = fem.form(a_20, entity_maps=entity_maps)
    a_22 = fem.form(a_22, entity_maps=entity_maps)

    L_0 = fem.form(inner(f + u_n / delta_t, v) * dx_c)
    L_1 = fem.form(inner(fem.Constant(msh, 0.0), q) * dx_c)
    L_2 = fem.form(inner(fem.Constant(
        facet_mesh, (PETSc.ScalarType(0.0),
                     PETSc.ScalarType(0.0))), vbar) * dx_f)

    # NOTE: Don't set pressure BC to avoid affecting conservation properties.
    # MUMPS seems to cope with the small nullspace
    L_3 = 0.0
    bcs = []
    for name, bc in boundary_conditions.items():
        id = boundaries[name]

        bc_func = fem.Function(Vbar)
        bc_func.interpolate(bc)

        # NOTE: Need to change this term for Neumann BCs
        L_3 += inner(dot(bc_func, n), qbar) * ds_c(id)

        facets = inv_entity_map[mt.indices[mt.values == id]]
        dofs = fem.locate_dofs_topological(Vbar, fdim, facets)
        bcs.append(fem.dirichletbc(bc_func, dofs))
    L_3 = fem.form(L_3, entity_maps=entity_maps)

    a = [[a_00, a_01, a_02, a_03],
         [a_10, None, None, None],
         [a_20, None, a_22, None],
         [a_30, None, None, None]]
    L = [L_0, L_1, L_2, L_3]

    if solver_type == SolverType.NAVIER_STOKES:
        A = fem.petsc.create_matrix_block(a)
    else:
        A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
        A.assemble()

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

    if scheme == Scheme.RW:
        u_vis = fem.Function(V)
    else:
        V_vis = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k + 1))
        u_vis = fem.Function(V_vis)
    u_vis.name = "u"
    p_h = fem.Function(Q)
    p_h.name = "p"
    ubar_h = fem.Function(Vbar)
    ubar_h.name = "ubar"
    pbar_h = fem.Function(Qbar)
    pbar_h.name = "pbar"

    u_file = io.VTXWriter(msh.comm, "u.bp", [u_vis._cpp_object])
    p_file = io.VTXWriter(msh.comm, "p.bp", [p_h._cpp_object])
    ubar_file = io.VTXWriter(msh.comm, "ubar.bp", [ubar_h._cpp_object])
    pbar_file = io.VTXWriter(msh.comm, "pbar.bp", [pbar_h._cpp_object])

    u_file.write(0.0)
    p_file.write(0.0)
    ubar_file.write(0.0)
    pbar_file.write(0.0)

    t = 0.0
    for n in range(num_time_steps):
        t += delta_t.value

        if solver_type == SolverType.NAVIER_STOKES:
            A.zeroEntries()
            fem.petsc.assemble_matrix_block(A, a, bcs=bcs)
            A.assemble()

        with b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

        # Compute solution
        ksp.solve(b, x)

        u_offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        p_offset = u_offset + \
            Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
        ubar_offset = \
            p_offset + Vbar.dofmap.index_map.size_local * \
            Vbar.dofmap.index_map_bs
        u_n.x.array[:u_offset] = x.array_r[:u_offset]
        u_n.x.scatter_forward()
        p_h.x.array[:p_offset - u_offset] = x.array_r[u_offset:p_offset]
        p_h.x.scatter_forward()
        ubar_h.x.array[:ubar_offset -
                       p_offset] = x.array_r[p_offset:ubar_offset]
        ubar_h.x.scatter_forward()
        pbar_h.x.array[:(len(x.array_r) - ubar_offset)
                       ] = x.array_r[ubar_offset:]
        pbar_h.x.scatter_forward()

        u_vis.interpolate(u_n)

        u_file.write(t)
        p_file.write(t)
        ubar_file.write(t)
        pbar_file.write(t)

    e_div_u = norm_L2(msh.comm, div(u_n))
    e_jump_u = normal_jump_error(msh, u_n)
    par_print(f"e_div_u = {e_div_u}")
    par_print(f"e_jump_u = {e_jump_u}")

    x = ufl.SpatialCoordinate(msh)
    xbar = ufl.SpatialCoordinate(facet_mesh)
    if u_e is not None:
        e_u = norm_L2(msh.comm, u_n - u_e(x))
        e_ubar = norm_L2(msh.comm, ubar_h - u_e(xbar))
        par_print(f"e_u = {e_u}")
        par_print(f"e_ubar = {e_ubar}")

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
    def create_mesh(self):
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
    def create_mesh(self):
        def gaussian(x, a, sigma, mu):
            return a * np.exp(- 1 / 2 * ((x - mu) / sigma)**2)

        comm = MPI.COMM_WORLD
        gdim = 2

        gmsh.initialize()
        if comm.rank == 0:
            # TODO Pass options
            gmsh.model.add("gaussian_bump")
            lc = 0.1
            a = 0.2
            sigma = 0.2
            mu = 1.0
            w = 5.0
            order = 1
            num_bottom_points = 100

            # Point tags
            bottom_points = [
                gmsh.model.geo.addPoint(x, gaussian(x, a, sigma, mu), 0.0, lc)
                for x in np.linspace(0.0, w, num_bottom_points)]
            top_left_point = gmsh.model.geo.addPoint(0, 1, 0, lc)
            top_right_point = gmsh.model.geo.addPoint(w, 1, 0, lc)

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
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)
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
        def u_d_lr(x): return np.vstack(
            (5.0 * x[1] * (1 - x[1]), np.zeros_like(x[0])))

        def u_d_tb(x): return np.vstack(
            (np.zeros_like(x[0]), np.zeros_like(x[0])))

        return {"left": u_d_lr,
                "right": u_d_lr,
                "bottom": u_d_tb,
                "top": u_d_tb}

    def f(self, msh):
        return fem.Constant(msh, (PETSc.ScalarType(0.0),
                                  PETSc.ScalarType(0.0)))


class Square(Problem):
    def create_mesh(self):
        comm = MPI.COMM_WORLD
        gdim = 2

        gmsh.initialize()
        if comm.rank == 0:
            gmsh.model.add("unit_square")
            lc = 0.1

            # Point tags
            points = [gmsh.model.geo.addPoint(0, 0, 0, lc),
                      gmsh.model.geo.addPoint(1, 0, 0, lc),
                      gmsh.model.geo.addPoint(1, 1, 0, lc),
                      gmsh.model.geo.addPoint(0, 1, 0, lc)]

            # Line tags
            lines = [gmsh.model.geo.addLine(points[0], points[1]),
                     gmsh.model.geo.addLine(points[1], points[2]),
                     gmsh.model.geo.addLine(points[2], points[3]),
                     gmsh.model.geo.addLine(points[3], points[0])]

            gmsh.model.geo.addCurveLoop(lines, 1)
            gmsh.model.geo.addPlaneSurface([1], 1)
            gmsh.model.geo.synchronize()

            gmsh.model.addPhysicalGroup(2, [1], 1)

            gmsh.model.addPhysicalGroup(1, [lines[0]], 1)
            gmsh.model.addPhysicalGroup(1, [lines[1]], 2)
            gmsh.model.addPhysicalGroup(1, [lines[2]], 3)
            gmsh.model.addPhysicalGroup(1, [lines[3]], 4)

            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.model.mesh.generate(2)

        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
        msh, _, mt = gmshio.model_to_mesh(
            gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
        gmsh.finalize()

        boundaries = {"left": 4,
                      "right": 2,
                      "bottom": 1,
                      "top": 3}
        return msh, mt, boundaries

    def u_e(self, x, module=ufl):
        u_x = module.sin(module.pi * x[0]) * module.sin(module.pi * x[1])
        u_y = module.cos(module.pi * x[0]) * module.cos(module.pi * x[1])
        if module == ufl:
            return ufl.as_vector((u_x, u_y))
        else:
            assert module == np
            return np.vstack((u_x, u_y))

    def p_e(self, x, module=ufl):
        return module.sin(module.pi * x[0]) * module.cos(module.pi * x[1])

    def boundary_conditions(self):
        def u_bc(x): return self.u_e(x, module=np)
        return {"left": u_bc,
                "right": u_bc,
                "bottom": u_bc,
                "top": u_bc}

    def f(self, msh):
        x = ufl.SpatialCoordinate(msh)
        f = - nu * div(grad(self.u_e(x))) + grad(self.p_e(x))
        if solver_type == SolverType.NAVIER_STOKES:
            f += div(outer(self.u_e(x), self.u_e(x)))
        return f


if __name__ == "__main__":
    # Simulation parameters
    solver_type = SolverType.NAVIER_STOKES
    k = 2
    nu = 1.0e-2
    num_time_steps = 10
    delta_t = 0.1
    scheme = Scheme.DRW

    comm = MPI.COMM_WORLD
    problem = Square()
    msh, mt, boundaries = problem.create_mesh()
    boundary_conditions = problem.boundary_conditions()
    f = problem.f(msh)

    solve(solver_type, k, nu, num_time_steps,
          delta_t, scheme, msh, mt, boundaries,
          boundary_conditions, f, problem.u_e,
          problem.p_e)
