from dolfinx import fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from ufl import (TrialFunction, TestFunction, CellDiameter, FacetNormal,
                 inner, grad, dx, dS, avg, outer, div, conditional,
                 gt, dot, Measure, as_vector)
from ufl import jump as jump_T
import gmsh
from utils import convert_facet_tags


def generate_mesh(comm, h=0.1, h_fac=1/3):
    gmsh.initialize()

    volume_id = {"fluid": 1,
                 "solid": 2}

    boundary_id = {"left": 2,
                   "right": 3,
                   "bottom": 4,
                   "top": 5,
                   "obstacle": 6}
    gdim = 2

    if comm.rank == 0:

        gmsh.model.add("model")
        factory = gmsh.model.geo

        length = 1
        height = 2
        c = (0.49, 0.5)
        r = 0.05

        rectangle_points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(length, 0.0, 0.0, h),
            factory.addPoint(length, height, 0.0, h),
            factory.addPoint(0.0, height, 0.0, h)
        ]

        circle_points = [
            factory.addPoint(c[0], c[1], 0.0, h),
            factory.addPoint(c[0] + r, c[1], 0.0, h * h_fac),
            factory.addPoint(c[0], c[1] + r, 0.0, h * h_fac),
            factory.addPoint(c[0] - r, c[1], 0.0, h * h_fac),
            factory.addPoint(c[0], c[1] - r, 0.0, h * h_fac)
        ]

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

        rectangle_curve = factory.addCurveLoop(rectangle_lines)
        circle_curve = factory.addCurveLoop(circle_lines)

        square_surface = factory.addPlaneSurface(
            [rectangle_curve, circle_curve])
        circle_surface = factory.addPlaneSurface([circle_curve])

        factory.synchronize()

        gmsh.model.addPhysicalGroup(2, [square_surface], volume_id["fluid"])
        gmsh.model.addPhysicalGroup(2, [circle_surface], volume_id["solid"])

        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[0]], boundary_id["bottom"])
        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[1]], boundary_id["right"])
        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[2]], boundary_id["top"])
        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[3]], boundary_id["left"])
        gmsh.model.addPhysicalGroup(1, circle_lines, boundary_id["obstacle"])

        gmsh.model.mesh.generate(2)

        # gmsh.fltk.run()

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, ct, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
    ft.name = "Facet markers"

    return msh, ct, ft, volume_id, boundary_id

# We also define some helper functions that will be used later


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


# We define some simulation parameters

num_time_steps = 50
t_end = 3
R_e = 1e6  # Reynolds Number
h = 0.05
h_fac = 1 / 3  # Factor scaling h near the cylinder
k = 2  # Polynomial degree

comm = MPI.COMM_WORLD

# Next, we create a mesh and the required functions spaces over
# it. Since the velocity uses an $H(\textnormal{div})$-conforming function
# space, we also create a vector valued discontinuous Lagrange space
# to interpolate into for artifact free visualisation.

msh, ct, ft, volume_id, boundary_id = generate_mesh(comm, h=h, h_fac=h_fac)

# with io.XDMFFile(msh.comm, "mesh.xdmf", "w") as file:
#     file.write_mesh(msh)
#     file.write_meshtags(ct)
#     file.write_meshtags(ft)

tdim = msh.topology.dim
submesh_f, entity_map_f = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == volume_id["fluid"]])[:2]
fdim = tdim - 1
submesh_f.topology.create_connectivity(fdim, tdim)
ft_f = convert_facet_tags(msh, submesh_f, entity_map_f, ft)
# with io.XDMFFile(msh.comm, "submesh_f.xdmf", "w") as file:
#     file.write_mesh(submesh_f)
#     file.write_meshtags(ft_f)

submesh_s, entity_map_s = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == volume_id["solid"]])[:2]

# Function space for the velocity
V = fem.FunctionSpace(submesh_f, ("Raviart-Thomas", k + 1))
# Function space for the pressure
Q = fem.FunctionSpace(submesh_f, ("Discontinuous Lagrange", k))
# Funcion space for visualising the velocity field
W = fem.VectorFunctionSpace(submesh_f, ("Discontinuous Lagrange", k + 1))

# Define trial and test functions

u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)
T, w = TrialFunction(Q), TestFunction(Q)

# delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
delta_t = t_end / num_time_steps  # TODO Make constant
# alpha = fem.Constant(msh, PETSc.ScalarType(6.0 * k**2))
alpha = 6.0 * k**2  # TODO Make constant
R_e_const = fem.Constant(submesh_f, PETSc.ScalarType(R_e))
kappa_f = fem.Constant(submesh_f, PETSc.ScalarType(0.001))

# List of tuples of form (id, expression)
dirichlet_bcs = [(boundary_id["bottom"],
                  lambda x: np.vstack((
                      np.zeros_like(x[0]), np.zeros_like(x[0])))),
                 (boundary_id["right"], lambda x: np.vstack(
                     (np.zeros_like(x[0]), np.zeros_like(x[0])))),
                 (boundary_id["top"],
                  lambda x: np.vstack((
                      np.zeros_like(x[0]), np.zeros_like(x[0])))),
                 (boundary_id["left"],
                  lambda x: np.vstack((
                      np.zeros_like(x[0]), np.zeros_like(x[0])))),
                 (boundary_id["obstacle"],
                  lambda x: np.vstack((
                      np.zeros_like(x[0]), np.zeros_like(x[0]))))]
neumann_bcs = []

ds_f = Measure("ds", domain=submesh_f, subdomain_data=ft_f)

h = CellDiameter(submesh_f)
n = FacetNormal(submesh_f)


def jump(phi, n):
    return outer(phi("+"), n("+")) + outer(phi("-"), n("-"))


# We solve the Stokes problem for the initial condition, so the convective
# terms are omitted for now

a_00 = 1 / R_e_const * (inner(grad(u), grad(v)) * dx
                        - inner(avg(grad(u)), jump(v, n)) * dS
                        - inner(jump(u, n), avg(grad(v))) * dS
                        + alpha / avg(h) * inner(jump(u, n), jump(v, n)) * dS)
a_01 = - inner(p, div(v)) * dx
a_10 = - inner(div(u), q) * dx

f = fem.Function(W)
# NOTE: Arrived at Neumann BC term by rewriting inner(grad(u), outer(v, n))
# it is based on as inner(dot(grad(u), n), v) and then g = dot(grad(u), n)
# etc. TODO Check this. NOTE Consider changing formulation to one with momentum
# law in conservative form to be able to specify momentum flux
L_0 = inner(f, v) * dx
L_1 = inner(fem.Constant(submesh_f, PETSc.ScalarType(0.0)), q) * dx

bcs = []
for bc in dirichlet_bcs:
    a_00 += 1 / R_e_const * (- inner(grad(u), outer(v, n)) * ds_f(bc[0])
                             - inner(outer(u, n), grad(v)) * ds_f(bc[0])
                             + alpha / h * inner(
        outer(u, n), outer(v, n)) * ds_f(bc[0]))
    u_D = fem.Function(V)
    u_D.interpolate(bc[1])
    L_0 += 1 / R_e_const * (- inner(outer(u_D, n), grad(v)) * ds_f(bc[0])
                            + alpha / h * inner(
                                outer(u_D, n), outer(v, n)) * ds_f(bc[0]))

    bc_boundary_facets = ft_f.indices[ft_f.values == bc[0]]
    bc_dofs = fem.locate_dofs_topological(
        V, submesh_f.topology.dim - 1, bc_boundary_facets)
    bcs.append(fem.dirichletbc(u_D, bc_dofs))


for bc in neumann_bcs:
    L_0 += 1 / R_e_const * inner(bc[1], v) * ds_f(bc[0])

a = fem.form([[a_00, a_01],
              [a_10, None]])
L = fem.form([L_0,
              L_1])

# Assemble Stokes problem

A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

# Create and configure solver

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()
# See https://graal.ens-lyon.fr/MUMPS/doc/userguide_5.5.1.pdf
# TODO Check
opts["mat_mumps_icntl_6"] = 2
opts["mat_mumps_icntl_14"] = 100
opts["ksp_error_if_not_converged"] = 1

if len(neumann_bcs) == 0:
    # Options to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_24"] = 1
    opts["mat_mumps_icntl_25"] = 0
ksp.setFromOptions()

# Solve Stokes for initial condition

x = A.createVecRight()
ksp.solve(b, x)

# Split the solution

u_h = fem.Function(V)
p_h = fem.Function(Q)
p_h.name = "p"
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u_h.x.array[:offset] = x.array_r[:offset]
u_h.x.scatter_forward()
p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
p_h.x.scatter_forward()
if len(neumann_bcs) == 0:
    p_h.x.array[:] -= domain_average(submesh_f, p_h)

u_vis = fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)

# Write initial condition to file

u_file = io.VTXWriter(msh.comm, "u.bp", [u_vis._cpp_object])
p_file = io.VTXWriter(msh.comm, "p.bp", [p_h._cpp_object])
t = 0.0
u_file.write(t)
p_file.write(t)

T_n = fem.Function(Q)

dirichlet_bcs_T = [(boundary_id["bottom"], lambda x: np.zeros_like(x[0])),
                   (boundary_id["right"], lambda x: np.zeros_like(x[0])),
                   (boundary_id["top"], lambda x: np.zeros_like(x[0])),
                   (boundary_id["left"], lambda x: np.zeros_like(x[0]))]

# Create function to store solution and previous time step

u_n = fem.Function(V)
u_n.x.array[:] = u_h.x.array

lmbda = conditional(gt(dot(u_n, n), 0), 1, 0)

# Create entity maps
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
inv_entity_map_f = np.full(num_cells, -1)
inv_entity_map_f[entity_map_f] = np.arange(len(entity_map_f))
inv_entity_map_s = np.full(num_cells, -1)
inv_entity_map_s[entity_map_s] = np.arange(len(entity_map_s))
entity_maps = {submesh_f: inv_entity_map_f,
               submesh_s: inv_entity_map_s}

dx_T = Measure("dx", domain=msh, subdomain_data=ct)
ds_T = Measure("ds", domain=msh, subdomain_data=ft)

# Create measure for integration. Assign the first (cell, local facet)
# pair to the cell in omega_0, corresponding to the "+" restriction. Assign
# the second pair to the omega_1 cell, corresponding to the "-" restriction.
fluid_int_facets = 7  # FIXME Don't hardcode
facet_integration_entities = {boundary_id["obstacle"]: [],
                              fluid_int_facets: []}
facet_imap = msh.topology.index_map(fdim)
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
domain_f_cells = ct.indices[ct.values == volume_id["fluid"]]
domain_s_cells = ct.indices[ct.values == volume_id["solid"]]
interface_facets = ft.indices[ft.values == boundary_id["obstacle"]]
for facet in interface_facets:
    # Check if this facet is owned
    if facet < facet_imap.size_local:
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        cell_plus = cells[0] if cells[0] in domain_f_cells else cells[1]
        cell_minus = cells[0] if cells[0] in domain_s_cells else cells[1]
        assert cell_plus in domain_f_cells
        assert cell_minus in domain_s_cells

        # FIXME Don't use tolist
        local_facet_plus = c_to_f.links(
            cell_plus).tolist().index(facet)
        local_facet_minus = c_to_f.links(
            cell_minus).tolist().index(facet)
        facet_integration_entities[boundary_id["obstacle"]].extend(
            [cell_plus, local_facet_plus, cell_minus, local_facet_minus])

        # HACK cell_minus does not exist in the left submesh, so it will
        # be mapped to index -1. This is problematic for the assembler,
        # which assumes it is possible to get the full macro dofmap for the
        # trial and test functions, despite the restriction meaning we
        # don't need the non-existant dofs. To fix this, we just map
        # cell_minus to the cell corresponding to cell plus. This will
        # just add zeros to the assembled system, since there are no
        # u("-") terms. Could map this to any cell in the submesh, but
        # I think using the cell on the other side of the facet means a
        # facet space coefficient could be used
        entity_maps[submesh_f][cell_minus] = \
            entity_maps[submesh_f][cell_plus]
        # Same hack for the right submesh
        entity_maps[submesh_s][cell_plus] = \
            entity_maps[submesh_s][cell_minus]

# FIXME Do this more efficiently
submesh_f.topology.create_entities(fdim)
submesh_f.topology.create_connectivity(tdim, fdim)
submesh_f.topology.create_connectivity(fdim, tdim)
c_to_f_submesh_f = submesh_f.topology.connectivity(tdim, fdim)
f_to_c_submesh_f = submesh_f.topology.connectivity(fdim, tdim)
for facet in range(submesh_f.topology.index_map(fdim).size_local):
    cells = f_to_c_submesh_f.links(facet)
    if len(cells) == 2:
        # FIXME Don't use tolist
        local_facet_plus = c_to_f_submesh_f.links(
            cells[0]).tolist().index(facet)
        local_facet_minus = c_to_f_submesh_f.links(
            cells[1]).tolist().index(facet)

        facet_integration_entities[fluid_int_facets].extend(
            [entity_map_f[cells[0]], local_facet_plus,
             entity_map_f[cells[1]], local_facet_minus])
dS_T = Measure("dS", domain=msh,
               subdomain_data=facet_integration_entities)

h_T = CellDiameter(msh)
n_T = FacetNormal(msh)
lmbda_T = conditional(gt(dot(u_n, n_T), 0), 1, 0)

gamma_int = 10  # Penalty param on interface
a_T_00 = inner(T / delta_t, w) * dx_T(volume_id["fluid"]) - \
    inner(u_h * T, grad(w)) * dx_T(volume_id["fluid"]) + \
    inner(lmbda_T("+") * dot(u_h("+"), n_T("+")) * T("+") -
          lmbda_T("-") * dot(u_h("-"), n_T("-")) * T("-"),
          jump_T(w)) * dS_T(fluid_int_facets) + \
    inner(lmbda_T * dot(u_h, n_T) * T, w) * ds_T + \
    kappa_f * (inner(grad(T), grad(w)) * dx_T(volume_id["fluid"]) -
               inner(avg(grad(T)), jump_T(w, n_T)) * dS_T(fluid_int_facets) -
               inner(jump_T(T, n_T), avg(grad(w))) * dS_T(fluid_int_facets) +
               (alpha / avg(h_T)) * inner(jump_T(T, n_T),
               jump_T(w, n_T)) * dS_T(fluid_int_facets)) \
    + kappa_f * (gamma_int / avg(h_T) * inner(
        T("+"), w("+")) * dS_T(boundary_id["obstacle"])
    - inner(1 / 2 * dot(grad(T("+")), n_T("+")),
            w("+")) * dS_T(boundary_id["obstacle"])
    - inner(1 / 2 * dot(grad(w("+")), n_T("+")),
            T("+")) * dS_T(boundary_id["obstacle"]))

L_T_0 = inner(T_n / delta_t, w) * dx_T(volume_id["fluid"])

for bc in dirichlet_bcs_T:
    T_D = fem.Function(Q)
    T_D.interpolate(bc[1])
    a_T_00 += kappa_f * (- inner(grad(T), w * n_T) * ds_T(bc[0]) -
                         inner(grad(w), T * n_T) * ds_T(bc[0]) +
                         (alpha / h_T) * inner(T, w) * ds_T(bc[0]))
    L_T_0 += - inner((1 - lmbda_T) * dot(u_h, n_T) * T_D, w) * ds_T(bc[0]) + \
        kappa_f * (- inner(T_D * n_T, grad(w)) * ds_T(bc[0]) +
                   (alpha / h_T) * inner(T_D, w) * ds_T(bc[0]))

a_T = fem.form(a_T_00, entity_maps=entity_maps)
L_T = fem.form(L_T_0, entity_maps=entity_maps)

A_T = fem.petsc.create_matrix(a_T)
b_T = fem.petsc.create_vector(L_T)

ksp_T = PETSc.KSP().create(msh.comm)
ksp_T.setOperators(A_T)
ksp_T.setType("preonly")
ksp_T.getPC().setType("lu")
ksp_T.getPC().setFactorSolverType("superlu_dist")

T_file = io.VTXWriter(msh.comm, "T.bp", [T_n._cpp_object])
T_file.write(t)

# Now we add the time stepping, convective, and buoyancy terms
# TODO Figure out correct way of "linearising"
# For buoyancy term, see
# https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy)
# where I've omitted the rho g h part (can think of this is
# lumping gravity in with pressure, see 2P4 notes) and taken
# T_0 to be 0
g = as_vector((0.0, -9.81))
rho_0 = fem.Constant(submesh_f, PETSc.ScalarType(1.0))
# Thermal expansion coeff
eps = fem.Constant(submesh_f, PETSc.ScalarType(10.0))

u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")
a_00 += inner(u / delta_t, v) * dx - \
    inner(u, div(outer(v, u_n))) * dx + \
    inner((dot(u_n, n))("+") * u_uw, v("+")) * dS + \
    inner((dot(u_n, n))("-") * u_uw, v("-")) * dS + \
    inner(dot(u_n, n) * lmbda * u, v) * ds_f
a = fem.form([[a_00, a_01],
              [a_10, None]])

L_0 += inner(u_n / delta_t - eps * rho_0 * T_n * g, v) * dx

for bc in dirichlet_bcs:
    u_D = fem.Function(V)
    u_D.interpolate(bc[1])
    L_0 += - inner(dot(u_n, n) * (1 - lmbda) * u_D, v) * ds_f(bc[0])
L = fem.form([L_0,
              L_1])

# Time stepping loop

for n in range(num_time_steps):
    # t += delta_t.value
    t += delta_t

    A.zeroEntries()
    fem.petsc.assemble_matrix_block(A, a, bcs=bcs)
    A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)
    u_h.x.array[:offset] = x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()
    if len(neumann_bcs) == 0:
        p_h.x.array[:] -= domain_average(submesh_f, p_h)

    A_T.zeroEntries()
    fem.petsc.assemble_matrix(A_T, a_T)
    A_T.assemble()

    with b_T.localForm() as b_T_loc:
        b_T_loc.set(0)
    fem.petsc.assemble_vector(b_T, L_T)
    b_T.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp_T.solve(b_T, T_n.vector)
    T_n.x.scatter_forward()

    u_vis.interpolate(u_h)

    # Write to file
    u_file.write(t)
    p_file.write(t)
    T_file.write(t)

    # Update u_n
    u_n.x.array[:] = u_h.x.array

u_file.close()
p_file.close()

# Compute errors
e_div_u = norm_L2(msh.comm, div(u_h))
# This scheme conserves mass exactly, so check this
assert np.isclose(e_div_u, 0.0)
