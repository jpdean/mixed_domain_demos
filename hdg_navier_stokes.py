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


class SolverType(Enum):
    STOKES = 1
    NAVIER_STOKES = 2


class Scheme(Enum):
    RW = 1
    DRW = 2


# Simulation parameters
solver_type = SolverType.NAVIER_STOKES
n = 32
k = 2
nu = 1.0e-2
num_time_steps = 10
delta_t = 10
scheme = Scheme.DRW


def u_e(x):
    return ufl.as_vector(
        (ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
         ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])))


def p_e(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])


def boundary(x):
    lr = np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    tb = np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    return np.logical_or(lr, tb)


comm = MPI.COMM_WORLD
rank = comm.rank
out_str = f"rank {rank}:\n"

msh = mesh.create_unit_square(
    comm, n, n, ghost_mode=mesh.GhostMode.none,
    cell_type=mesh.CellType.quadrilateral)

tdim = msh.topology.dim
fdim = tdim - 1

num_cell_facets = cell_num_entities(msh.topology.cell_type, fdim)
msh.topology.create_entities(fdim)
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)

# NOTE Despite all facets being present in the submesh, the entity map isn't
# necessarily the identity in parallel
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

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
gamma = 6.0 * k**2 / h

boundaries = {"exterior": 2}

msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
values = np.full_like(
    msh_boundary_facets, boundaries["exterior"], dtype=np.intc)
mt = mesh.meshtags(msh, fdim, msh_boundary_facets, values)

facet_integration_entities = compute_integration_domains(
    fem.IntegralType.exterior_facet, mt)

all_facets = np.amax(values) + 1
facet_integration_entities[all_facets] = []
for cell in range(msh.topology.index_map(tdim).size_local):
    for local_facet in range(num_cell_facets):
        facet_integration_entities[all_facets].extend([cell, local_facet])

dx_c = ufl.Measure("dx", domain=msh)
ds_c = ufl.Measure("ds", subdomain_data=facet_integration_entities, domain=msh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

inv_entity_map = np.full_like(entity_map, -1)
for i, f in enumerate(entity_map):
    inv_entity_map[f] = i
entity_maps = {facet_mesh: inv_entity_map}

u_d = fem.Function(Vbar)
u_d_expr = fem.Expression(u_e(ufl.SpatialCoordinate(facet_mesh)),
                          Vbar.element.interpolation_points())
u_d.interpolate(u_d_expr)

x = ufl.SpatialCoordinate(msh)
f = - nu * div(grad(u_e(x))) + grad(p_e(x))
if solver_type == SolverType.NAVIER_STOKES:
    f += div(outer(u_e(x), u_e(x)))
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
a_03 = fem.form(inner(dot(v, n), pbar) * ds_c(all_facets), entity_maps=entity_maps)
a_10 = fem.form(- inner(q, div(u)) * dx_c)
a_20 = nu * (inner(vbar, dot(grad(u), n)) * ds_c(all_facets)
             - gamma * inner(vbar, u) * ds_c(all_facets))
a_30 = fem.form(inner(dot(u, n), qbar) * ds_c(all_facets), entity_maps=entity_maps)
a_22 = nu * gamma * inner(ubar, vbar) * ds_c(all_facets)

if solver_type == SolverType.NAVIER_STOKES:
    a_00 += inner(outer(u, u_n) - outer(u, lmbda * u_n),
                  outer(v, n)) * ds_c(all_facets) - \
        inner(outer(u, u_n), grad(v)) * dx_c
    a_02 += inner(outer(ubar, lmbda * u_n), outer(v, n)) * ds_c(all_facets)
    a_20 += inner(outer(u, u_n) - outer(u, lmbda * u_n),
                  outer(vbar, n)) * ds_c(all_facets)
    a_22 += inner(outer(ubar, lmbda * u_n), outer(vbar, n)) * ds_c(all_facets)

a_00 = fem.form(a_00)
a_02 = fem.form(a_02, entity_maps=entity_maps)
a_20 = fem.form(a_20, entity_maps=entity_maps)
a_22 = fem.form(a_22, entity_maps=entity_maps)

L_0 = fem.form(inner(f + u_n / delta_t, v) * dx_c)
L_1 = fem.form(inner(fem.Constant(msh, 0.0), q) * dx_c)
L_2 = fem.form(inner(fem.Constant(
    facet_mesh, (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0))), vbar) * dx_f)
# NOTE: Need to change this term for Neumann BCs
L_3 = fem.form(inner(dot(u_d, n), qbar) * ds_c(2), entity_maps=entity_maps)

a = [[a_00, a_01, a_02, a_03],
     [a_10, None, None, None],
     [a_20, None, a_22, None],
     [a_30, None, None, None]]
L = [L_0, L_1, L_2, L_3]

facet_mesh_boundary_facets = inv_entity_map[msh_boundary_facets]
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
bc_ubar = fem.dirichletbc(u_d, dofs)

# NOTE: Don't set pressure BC to avoid affecting conservation properties.
# MUMPS seems to cope with the small nullspace
bcs = [bc_ubar]

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
    p_offset = u_offset + Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    ubar_offset = \
        p_offset + Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
    u_n.x.array[:u_offset] = x.array_r[:u_offset]
    u_n.x.scatter_forward()
    p_h.x.array[:p_offset - u_offset] = x.array_r[u_offset:p_offset]
    p_h.x.scatter_forward()
    ubar_h.x.array[:ubar_offset - p_offset] = x.array_r[p_offset:ubar_offset]
    ubar_h.x.scatter_forward()
    pbar_h.x.array[:(len(x.array_r) - ubar_offset)] = x.array_r[ubar_offset:]
    pbar_h.x.scatter_forward()

    u_vis.interpolate(u_n)

    u_file.write(t)
    p_file.write(t)
    ubar_file.write(t)
    pbar_file.write(t)

x = ufl.SpatialCoordinate(msh)
e_u = norm_L2(msh.comm, u_n - u_e(x))
e_div_u = norm_L2(msh.comm, div(u_n))
e_jump_u = normal_jump_error(msh, u_n)
p_h_avg = domain_average(msh, p_h)
p_e_avg = domain_average(msh, p_e(x))
e_p = norm_L2(msh.comm, (p_h - p_h_avg) - (p_e(x) - p_e_avg))

xbar = ufl.SpatialCoordinate(facet_mesh)
e_ubar = norm_L2(msh.comm, ubar_h - u_e(xbar))
pbar_h_avg = domain_average(facet_mesh, pbar_h)
pbar_e_avg = domain_average(facet_mesh, p_e(xbar))
e_pbar = norm_L2(msh.comm, (pbar_h - pbar_h_avg) - (p_e(xbar) - pbar_e_avg))

if rank == 0:
    print(f"e_u = {e_u}")
    print(f"e_div_u = {e_div_u}")
    print(f"e_jump_u = {e_jump_u}")
    print(f"e_p = {e_p}")
    print(f"e_ubar = {e_ubar}")
    print(f"e_pbar = {e_pbar}")
