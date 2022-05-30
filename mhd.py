# Simplified version of fully coupled scheme from Hiptmair2018 eq. (3.5) but
# uses Stokes instead of Navier-Stokes. NOTE: Extra terms arise due to
# prescribed current density that Hiptmair doesn't include

from dolfinx import mesh, fem, graph
from mpi4py import MPI
from ufl import (TrialFunction, TestFunction, inner, grad, div,
                 curl, cross, Measure, dx)
from petsc4py import PETSc
import numpy as np
from dolfinx.io import VTXWriter, XDMFFile


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


def solve_mhd(msh, submesh, k, boundary_marker_msh, boundary_marker_submesh,
              f_expr, u_bc_expr, t_end, num_time_steps, A_expr, J_p_expr,
              entity_map):

    V = fem.VectorFunctionSpace(submesh, ("Lagrange", k))
    Q = fem.FunctionSpace(submesh, ("Lagrange", k - 1))

    # H(curl; Ω)-conforming space for magnetic vector potential and electric
    # field
    X = fem.FunctionSpace(msh, ("Nedelec 1st kind H(curl)", k + 1))
    # H(div; Ω)-conforming space for the magnetic flux density
    # Y = fem.FunctionSpace(msh, ("Raviart-Thomas", k))
    # [L^2(Ω)]^d-conforming function space for visualisation.
    Z = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k + 1))

    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define weak form of problem
    A = TrialFunction(X)
    phi = TestFunction(X)

    # Magnetic vector potential
    A_h = fem.Function(X)
    A_h.interpolate(A_expr)
    # Magnetic vector potential at previous time step
    A_n = fem.Function(X)
    A_n.x.array[:] = A_h.x.array

    # Prescribed current density
    J_p = fem.Function(Z)
    J_p.interpolate(J_p_expr)

    u_n = fem.Function(V)

    dx_sm = Measure("dx", domain=submesh)

    entity_maps_sm = {msh: entity_map}

    # TODO MAKE CONSTANT
    # delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
    delta_t = t_end / num_time_steps
    a_00 = fem.form(inner(u / delta_t, v) * dx_sm
                    + inner(grad(u), grad(v)) * dx_sm
                    + inner(cross(curl(A_n), u), cross(curl(A_n), v)) * dx_sm,
                    entity_maps=entity_maps_sm)
    a_01 = fem.form(- inner(p, div(v)) * dx_sm)
    a_02 = fem.form(inner(A / delta_t, cross(curl(A_n), v)) * dx_sm,
                    entity_maps=entity_maps_sm)
    a_10 = fem.form(- inner(div(u), q) * dx_sm)
    a_11 = fem.form(fem.Constant(submesh, PETSc.ScalarType(0.0))
                    * inner(p, q) * dx_sm)
    a_20 = fem.form(inner(cross(curl(A_n), u), phi) * dx_sm,
                    entity_maps=entity_maps_sm)
    a_22 = fem.form(inner(A / delta_t, phi) * dx
                    + inner(curl(A), curl(phi)) * dx)

    f = fem.Function(V)
    f.interpolate(f_expr)
    L_0 = fem.form(inner(u_n / delta_t + f, v) * dx_sm
                   + inner(A_n / delta_t + J_p, cross(curl(A_n), v)) * dx_sm,
                   entity_maps=entity_maps_sm)
    L_1 = fem.form(inner(fem.Constant(submesh, PETSc.ScalarType(0.0)), q) * dx_sm)
    L_2 = fem.form(inner(A_n / delta_t, phi) * dx + inner(J_p, phi) * dx)

    a = [[a_00, a_01, a_02],
         [a_10, a_11, None],
         [a_20, None, a_22]]

    L = [L_0,
         L_1,
         L_2]

    u_bc = fem.Function(V)
    u_bc.interpolate(u_bc_expr)
    boundary_facets_sm = mesh.locate_entities_boundary(
        submesh, submesh.topology.dim - 1, boundary_marker_sm)
    boundary_vel_dofs = fem.locate_dofs_topological(
        V, submesh.topology.dim - 1, boundary_facets_sm)
    bc_u = fem.dirichletbc(u_bc, boundary_vel_dofs)

    pressure_dof = fem.locate_dofs_geometrical(
        Q, lambda x: np.logical_and(np.logical_and(np.isclose(x[0], 0.0),
                                                   np.isclose(x[1], 0.0)),
                                    np.isclose(x[2], 0.0)))
    bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), pressure_dof, Q)

    boundary_facets_m = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, boundary_marker_msh)
    boundary_A_dofs = fem.locate_dofs_topological(
        X, msh.topology.dim - 1, boundary_facets_m)
    A_bc = fem.Function(X)
    A_bc.interpolate(A_expr)
    bc_A = fem.dirichletbc(A_bc, boundary_A_dofs)

    bcs = [bc_u, bc_p, bc_A]

    # FIXME Just create matrix and vector here
    A_mat = fem.petsc.assemble_matrix_block(a, bcs=bcs)
    A_mat.assemble()
    b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A_mat)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")

    # Compute solution
    x = A_mat.createVecLeft()

    # Create Functions and scatter x solution
    u, p = fem.Function(V), fem.Function(Q)

    A_Z = fem.Function(Z)
    A_Z.name = "A"
    A_Z.interpolate(A_h)

    # TODO Write in one file
    u_file = VTXWriter(submesh.comm, "u.bp", [u._cpp_object])
    p_file = VTXWriter(submesh.comm, "p.bp", [p._cpp_object])
    A_file = VTXWriter(msh.comm, "A.bp", [A_Z._cpp_object])

    t = 0.0
    u_file.write(t)
    p_file.write(t)
    A_file.write(t)
    offset_0 = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    offset_1 = offset_0 + Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    for n in range(num_time_steps):
        t += delta_t
        # t += delta_t.value

        u_bc_expr.t = t
        u_bc.interpolate(u_bc_expr)

        A_expr.t = t
        A_bc.interpolate(A_expr)

        f_expr.t = t
        f.interpolate(f_expr)

        J_p_expr.t = t
        J_p.interpolate(J_p_expr)

        A_mat.zeroEntries()
        fem.petsc.assemble_matrix_block(A_mat, a, bcs=bcs)
        A_mat.assemble()

        with b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

        ksp.solve(b, x)
        u.x.array[:offset_0] = x.array_r[:offset_0]
        u.x.scatter_forward()
        p.x.array[:offset_1 - offset_0] = x.array_r[offset_0:offset_1]
        p.x.scatter_forward()
        A_h.x.array[:(len(x.array_r) - offset_1)] = x.array_r[offset_1:]
        A_h.x.scatter_forward()

        A_Z.interpolate(A_h)

        u_file.write(t)
        p_file.write(t)
        A_file.write(t)

        u_n.x.array[:] = u.x.array
        A_n.x.array[:] = A_h.x.array

    u_file.close()
    p_file.close()
    A_file.close()

    return u, p, A_h


if __name__ == "__main__":
    def boundary_marker_msh(x):
        b_0 = np.logical_or(np.isclose(x[0], 0.0),
                            np.isclose(x[0], 1.0))
        b_1 = np.logical_or(np.isclose(x[1], 0.0),
                            np.isclose(x[1], 2.0))
        b_2 = np.logical_or(np.isclose(x[2], 0.0),
                            np.isclose(x[2], 1.0))
        return np.logical_or(np.logical_or(b_0, b_1), b_2)

    def boundary_marker_sm(x):
        b_0 = np.logical_or(np.isclose(x[0], 0.0),
                            np.isclose(x[0], 1.0))
        b_1 = np.logical_or(np.isclose(x[1], 0.0),
                            np.isclose(x[1], 1.0))
        b_2 = np.logical_or(np.isclose(x[2], 0.0),
                            np.isclose(x[2], 1.0))
        return np.logical_or(np.logical_or(b_0, b_1), b_2)

    class TimeDependentExpression:
        """Simple class to represent time dependent functions"""

        def __init__(self, expression):
            self.t = 0.0
            self.expression = expression

        def __call__(self, x):
            return self.expression(x, self.t)

    # NOTE n must be even
    n = 4
    k = 2
    t_end = 0.1
    num_time_steps = 5

    # Analytical solution from sol_gen.py
    u_expr = TimeDependentExpression(
        lambda x, t:
            np.vstack(
                (np.sin(x[1]*np.pi)*np.sin(np.pi*t),
                 np.zeros_like(x[0]),
                 np.zeros_like(x[0]))))

    p_expr = TimeDependentExpression(
        lambda x, t: np.sin(x[0]*np.pi)*np.sin(x[1]*np.pi)*np.sin(x[2]*np.pi)*np.sin(np.pi*t))

    f_expr = TimeDependentExpression(
        lambda x, t:
            np.vstack(
                (np.pi*(-np.pi**2*np.sin(x[0]*np.pi)*np.sin(np.pi*t)**2*np.cos(x[0]*np.pi) + np.sin(x[1]*np.pi)*np.sin(x[2]*np.pi)*np.sin(np.pi*t)*np.cos(x[0]*np.pi) + np.pi*np.sin(x[1]*np.pi)*np.sin(np.pi*t) + np.sin(x[1]*np.pi)*np.cos(np.pi*t)),
                 np.pi*np.sin(x[0]*np.pi)*np.sin(x[2]*np.pi) *
                 np.sin(np.pi*t)*np.cos(x[1]*np.pi),
                 np.pi*np.sin(x[0]*np.pi)*np.sin(x[1]*np.pi)*np.sin(np.pi*t)*np.cos(x[2]*np.pi))))

    A_expr = TimeDependentExpression(
        expression=lambda x, t:
        np.vstack((np.zeros_like(x[0]),
                   np.sin(x[0]*np.pi)*np.sin(np.pi*t),
                   np.zeros_like(x[0]))))

    J_p_expr = TimeDependentExpression(
        expression=lambda x, t:
        np.vstack((np.zeros_like(x[0]),
                   np.pi**2*np.sin(x[0]*np.pi)*np.sin(np.pi*t) + np.pi*np.sin(x[0]*np.pi)*np.cos(
                       np.pi*t) + np.pi*np.sin(x[1]*np.pi)*np.sin(np.pi*t)**2*np.cos(x[0]*np.pi),
                   np.zeros_like(x[0]))))

    cube_msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    msh = mesh.create_box(
        MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (1.0, 2.0, 1.0)), (n, 2 * n, n))
    upper_cells = mesh.locate_entities(
        msh, msh.topology.dim, lambda x: x[1] <= 1)
    submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(
        msh, msh.topology.dim, upper_cells)
    with XDMFFile(msh.comm, "msh.xdmf", "w") as file:
        file.write_mesh(msh)
    with XDMFFile(cube_msh.comm, "cube_msh.xdmf", "w") as file:
        file.write_mesh(cube_msh)
    with XDMFFile(submesh.comm, "submesh.xdmf", "w") as file:
        file.write_mesh(submesh)

    u_h, p_h, A_h = solve_mhd(
        msh, submesh, k, boundary_marker_msh, boundary_marker_sm, f_expr, u_expr,
        t_end, num_time_steps, A_expr, J_p_expr, entity_map)

    # V = fem.VectorFunctionSpace(submesh, ("Discontinuous Lagrange", k + 3))
    # Q = fem.FunctionSpace(submesh, ("Lagrange", k + 2))

    # u = fem.Function(V)
    # u_expr.t = t_end
    # u.interpolate(u_expr)

    # p = fem.Function(Q)
    # p_expr.t = t_end
    # p.interpolate(p_expr)

    # A = fem.Function(V)
    # A_expr.t = t_end
    # A.interpolate(A_expr)

    # e_u = norm_L2(submesh.comm, u_h - u)
    # e_div_u = norm_L2(submesh.comm, div(u_h))
    # p_h_avg = domain_average(submesh, p_h)
    # p_e_avg = domain_average(submesh, p)
    # e_p = norm_L2(submesh.comm, (p_h - p_h_avg) - (p - p_e_avg))
    # e_A = norm_L2(submesh.comm, A - A_h)

    # if msh.comm.Get_rank() == 0:
    #     print(f"e_u = {e_u}")
    #     print(f"e_div_u = {e_div_u}")
    #     print(f"e_p = {e_p}")
    #     print(f"e_A = {e_A}")
