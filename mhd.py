# Simplified version of the fully coupled MHD scheme from Hiptmair2018 eq. (3.5).and
# Here we solve Stokes using a Taylor-Hood scheme instead of Navier-Stokes.
# NOTE: Extra terms arise due to prescribed current density that Hiptmair doesn't
# include

from dolfinx import mesh, fem
from mpi4py import MPI
from ufl import (TrialFunction, TestFunction, inner, grad, div,
                 curl, cross, Measure, dx)
from petsc4py import PETSc
import numpy as np
from dolfinx.io import VTXWriter
from utils import norm_L2, domain_average


def solve_mhd(k, msh, boundary_marker_msh, submesh, boundary_marker_submesh,
              entity_map, A_expr, u_bc_expr, J_p_expr, f_expr, t_end,
              num_time_steps):
    # H(curl; Ω)-conforming space for magnetic vector potential and electric
    # field
    X = fem.FunctionSpace(msh, ("Nedelec 1st kind H(curl)", k + 1))
    # [L^2(Ω)]^d-conforming function space for visualisation.
    Z = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k + 1))

    # H^1(Ω) and [L^2(Ω)]^d conforming subspaces for the velocity and pressure
    V = fem.VectorFunctionSpace(submesh, ("Lagrange", k))
    Q = fem.FunctionSpace(submesh, ("Lagrange", k - 1))

    # Trial and test functions for the magnetic vector potential
    A = TrialFunction(X)
    phi = TestFunction(X)

    # Define trial and test functions for velocity and pressure spaces
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Function to represent magnetic vector potential at current time step
    A_h = fem.Function(X)
    A_h.interpolate(A_expr)
    # Magnetic vector potential at previous time step
    A_n = fem.Function(X)
    A_n.x.array[:] = A_h.x.array
    # Velocity at previous time step
    u_n = fem.Function(V)

    # Prescribed current density
    J_p = fem.Function(Z)
    J_p.interpolate(J_p_expr)

    # Force on fluid
    f = fem.Function(V)
    f.interpolate(f_expr)

    # Create integration measures and entity maps
    dx_sm = Measure("dx", domain=submesh)
    entity_maps_sm = {msh: entity_map}

    # Define forms
    # TODO Check I'm not missing conductivity terms in solid region
    delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
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

    L_0 = fem.form(inner(u_n / delta_t + f, v) * dx_sm
                   + inner(A_n / delta_t + J_p, cross(curl(A_n), v)) * dx_sm,
                   entity_maps=entity_maps_sm)
    L_1 = fem.form(
        inner(fem.Constant(submesh, PETSc.ScalarType(0.0)), q) * dx_sm)
    L_2 = fem.form(inner(A_n / delta_t, phi) * dx + inner(J_p, phi) * dx)

    a = [[a_00, a_01, a_02],
         [a_10, a_11, None],
         [a_20, None, a_22]]

    L = [L_0,
         L_1,
         L_2]

    # Magnetic vector potential boundary condition
    boundary_facets_m = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, boundary_marker_msh)
    boundary_A_dofs = fem.locate_dofs_topological(
        X, msh.topology.dim - 1, boundary_facets_m)
    A_bc = fem.Function(X)
    A_bc.interpolate(A_expr)
    bc_A = fem.dirichletbc(A_bc, boundary_A_dofs)

    # Velocity boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(u_bc_expr)
    boundary_facets_sm = mesh.locate_entities_boundary(
        submesh, submesh.topology.dim - 1, boundary_marker_submesh)
    boundary_vel_dofs = fem.locate_dofs_topological(
        V, submesh.topology.dim - 1, boundary_facets_sm)
    bc_u = fem.dirichletbc(u_bc, boundary_vel_dofs)

    # Pressure boundary condition
    pressure_dof = fem.locate_dofs_geometrical(
        Q, lambda x: np.logical_and(np.logical_and(np.isclose(x[0], 0.0),
                                                   np.isclose(x[1], 0.0)),
                                    np.isclose(x[2], 0.0)))
    print(pressure_dof)
    bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), pressure_dof, Q)

    # Collect boundary conditions
    bcs = [bc_u, bc_p, bc_A]

    # Create matrix and vector
    A_mat = fem.petsc.create_matrix_block(a)
    b = fem.petsc.create_vector_block(L)

    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A_mat)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")

    # Create vector to store solution
    x = A_mat.createVecLeft()

    # Create functions to output to file
    u_h, p_h = fem.Function(V), fem.Function(Q)
    u_h.name = "u"
    p_h.name = "p"
    # Interpolate A_h into Z for artifact-free visualization
    A_Z = fem.Function(Z)
    A_Z.name = "A"
    A_Z.interpolate(A_h)
    # Magnetic flux density (curl of magnetic vector potential)
    B = curl(A_h)
    B_expr = fem.Expression(B, Z.element.interpolation_points())
    B_Z = fem.Function(Z)
    B_Z.interpolate(B_expr)

    u_file = VTXWriter(submesh.comm, "u.bp", [u_h._cpp_object])
    p_file = VTXWriter(submesh.comm, "p.bp", [p_h._cpp_object])
    A_file = VTXWriter(msh.comm, "A.bp", [A_Z._cpp_object])
    B_file = VTXWriter(msh.comm, "B.bp", [B_Z._cpp_object])

    t = 0.0
    u_file.write(t)
    p_file.write(t)
    A_file.write(t)
    B_file.write(t)
    offset_0 = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    offset_1 = offset_0 + Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    for n in range(num_time_steps):
        t += delta_t.value

        # Update expressions
        u_bc_expr.t = t
        u_bc.interpolate(u_bc_expr)
        A_expr.t = t
        A_bc.interpolate(A_expr)
        f_expr.t = t
        f.interpolate(f_expr)
        J_p_expr.t = t
        J_p.interpolate(J_p_expr)

        # Assemble matrix
        A_mat.zeroEntries()
        fem.petsc.assemble_matrix_block(A_mat, a, bcs=bcs)
        A_mat.assemble()

        # Assemble vector
        with b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

        # Solve and recover solution
        ksp.solve(b, x)
        u_h.x.array[:offset_0] = x.array_r[:offset_0]
        u_h.x.scatter_forward()
        p_h.x.array[:offset_1 - offset_0] = x.array_r[offset_0:offset_1]
        p_h.x.scatter_forward()
        A_h.x.array[:(len(x.array_r) - offset_1)] = x.array_r[offset_1:]
        A_h.x.scatter_forward()

        # Interpolate for file output
        A_Z.interpolate(A_h)
        B_Z.interpolate(B_expr)

        # Save to file
        u_file.write(t)
        p_file.write(t)
        A_file.write(t)
        B_file.write(t)

        # Update
        u_n.x.array[:] = u_h.x.array
        A_n.x.array[:] = A_h.x.array

    u_file.close()
    p_file.close()
    A_file.close()
    B_file.close()

    return u_h, p_h, A_h


if __name__ == "__main__":
    # Boundary marker for the mesh, over which the electromagnetism
    # problem will be solved
    def boundary_marker_msh(x):
        b_0 = np.logical_or(np.isclose(x[0], 0.0),
                            np.isclose(x[0], 1.0))
        b_1 = np.logical_or(np.isclose(x[1], 0.0),
                            np.isclose(x[1], 2.0))
        b_2 = np.logical_or(np.isclose(x[2], 0.0),
                            np.isclose(x[2], 1.0))
        return np.logical_or(np.logical_or(b_0, b_1), b_2)

    # Boundary marker for the submesh, over which the fluid problem
    # will be solved
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

    # Simulation parameters
    n = 4
    assert n % 2 == 0  # NOTE n must be even
    k = 2
    t_end = 0.1
    num_time_steps = 5

    # Boundary condition for the velocity field
    u_expr = TimeDependentExpression(
        lambda x, t:
            np.vstack(
                (np.zeros_like(x[0]),
                 np.zeros_like(x[0]),
                 np.zeros_like(x[0]))))

    # Fluid forcing term
    f_expr = TimeDependentExpression(
        lambda x, t:
            np.vstack(
                (np.zeros_like(x[0]),
                 np.zeros_like(x[0]),
                 np.zeros_like(x[0]))))

    # Initial and boundary condition for the magnetic vector potential
    A_expr = TimeDependentExpression(
        expression=lambda x, t:
            np.vstack(
                (np.zeros_like(x[0]),
                 np.zeros_like(x[0]),
                 np.zeros_like(x[0]))))

    # Prescribed current density
    J_p_expr = TimeDependentExpression(
        expression=lambda x, t:
        np.vstack((np.zeros_like(x[0]),
                   np.sin(np.pi * t) * np.cos(np.pi * x[0]),
                   np.zeros_like(x[0]))))

    # Create mesh and submesh
    msh = mesh.create_box(
        MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (1.0, 2.0, 1.0)), (n, 2 * n, n))
    fluid_cells = mesh.locate_entities(
        msh, msh.topology.dim, lambda x: x[1] <= 1)
    submesh, entity_map = mesh.create_submesh(
        msh, msh.topology.dim, fluid_cells)[0:2]

    u_h, p_h, A_h = solve_mhd(
        k, msh, boundary_marker_msh, submesh, boundary_marker_sm,
        entity_map, A_expr, u_expr, J_p_expr, f_expr, t_end,
        num_time_steps)

    u_h_norm = norm_L2(msh.comm, u_h)
    p_h_average = domain_average(submesh, p_h)
    p_h_norm = norm_L2(msh.comm, p_h - p_h_average)
    A_h_norm = norm_L2(msh.comm, A_h)

    if msh.comm.Get_rank() == 0:
        print(u_h_norm)
        print(p_h_norm)
        print(A_h_norm)
