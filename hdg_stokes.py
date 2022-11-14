from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div, outer
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from utils import reorder_mesh, norm_L2, domain_average, normal_jump_error


comm = MPI.COMM_WORLD
rank = comm.rank
out_str = f"rank {rank}:\n"

n = 32
msh = mesh.create_unit_square(
    comm, n, n, ghost_mode=mesh.GhostMode.none)

# Currently, permutations are not working in parallel, so reorder the
# mesh
reorder_mesh(msh)

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

k = 2
V = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k))
Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k - 1))
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

facet_integration_entities = {1: []}
for cell in range(msh.topology.index_map(tdim).size_local):
    for local_facet in range(num_cell_facets):
        facet_integration_entities[1].extend([cell, local_facet])

dx_c = ufl.Measure("dx", domain=msh)
ds_c = ufl.Measure("ds", subdomain_data=facet_integration_entities, domain=msh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

inv_entity_map = np.full_like(entity_map, -1)
for i, f in enumerate(entity_map):
    inv_entity_map[f] = i
entity_maps = {facet_mesh: inv_entity_map}


def u_e(x):
    return ufl.as_vector(
        (x[0]**2 * (1 - x[0])**2 * (2 * x[1] - 6 * x[1]**2 + 4 * x[1]**3),
         - x[1]**2 * (1 - x[1])**2 * (2 * x[0] - 6 * x[0]**2 + 4 * x[0]**3)))


def p_e(x):
    return x[0] * (1 - x[0])


x = ufl.SpatialCoordinate(msh)
nu = 1.0e-2
f = - nu * div(grad(u_e(x))) + grad(p_e(x)) + div(outer(u_e(x), u_e(x)))
u_n = fem.Function(V)
lmbda = ufl.conditional(ufl.lt(dot(u_n, n), 0), 1, 0)

delta_t = fem.Constant(msh, PETSc.ScalarType(10.0))
nu = fem.Constant(msh, PETSc.ScalarType(nu))
num_time_steps = 10

# TODO Double check convective terms
a_00 = fem.form(inner(u / delta_t, v) * dx_c +
                nu * (inner(grad(u), grad(v)) * dx_c +
                gamma * inner(u, v) * ds_c(1)
                - (inner(u, dot(grad(v), n))
                   + inner(v, dot(grad(u), n))) * ds_c(1))
                + inner(outer(u, u_n) - outer(u, lmbda * u_n),
                        outer(v, n)) * ds_c(1) -
                inner(outer(u, u_n), grad(v)) * dx_c)
a_01 = fem.form(- inner(p, div(v)) * dx_c)
a_02 = fem.form(nu * (inner(ubar, dot(grad(v), n)) * ds_c(1)
                - gamma * inner(ubar, v) * ds_c(1)) +
                inner(outer(ubar, lmbda * u_n), outer(v, n)) * ds_c(1),
                entity_maps=entity_maps)
a_03 = fem.form(inner(dot(v, n), pbar) * ds_c(1), entity_maps=entity_maps)
a_10 = fem.form(- inner(q, div(u)) * dx_c)
# Only needed to apply BC on pressure
a_11 = fem.form(fem.Constant(msh, 0.0) * inner(p, q) * dx_c)
a_20 = fem.form(nu * (inner(vbar, dot(grad(u), n)) * ds_c(1)
                - gamma * inner(vbar, u) * ds_c(1)) +
                inner(outer(u, u_n) - outer(u, lmbda * u_n),
                      outer(vbar, n)) * ds_c(1),
                entity_maps=entity_maps)
a_30 = fem.form(inner(dot(u, n), qbar) * ds_c(1), entity_maps=entity_maps)
a_22 = fem.form(nu * gamma * inner(ubar, vbar) * ds_c(1) +
                inner(outer(ubar, lmbda * u_n), outer(vbar, n)) * ds_c(1),
                entity_maps=entity_maps)

L_0 = fem.form(inner(f + u_n / delta_t, v) * dx_c)
L_1 = fem.form(inner(fem.Constant(msh, 0.0), q) * dx_c)
L_2 = fem.form(inner(fem.Constant(
    facet_mesh, (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0))), vbar) * dx_f)
L_3 = fem.form(inner(fem.Constant(facet_mesh, 0.0), qbar) * dx_f)

a = [[a_00, a_01, a_02, a_03],
     [a_10, a_11, None, None],
     [a_20, None, a_22, None],
     [a_30, None, None, None]]
L = [L_0, L_1, L_2, L_3]


def boundary(x):
    lr = np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    tb = np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    return np.logical_or(lr, tb)


msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
facet_mesh_boundary_facets = inv_entity_map[msh_boundary_facets]
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
bc_ubar = fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), dofs, Vbar)

# Pressure boundary condition
# TODO Locate on facet space or cell space?
# FIXME Change so it doesn't depend on diagonal direction
pressure_dof = fem.locate_dofs_geometrical(
    Q, lambda x: np.logical_and(np.isclose(x[0], 1.0),
                                np.isclose(x[1], 0.0)))
bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), pressure_dof, Q)

bcs = [bc_ubar, bc_p]

A = fem.petsc.create_matrix_block(a)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

b = fem.petsc.create_vector_block(L)
x = A.createVecRight()

u_n.name = "u"
p_h = fem.Function(Q)
p_h.name = "p"
ubar_h = fem.Function(Vbar)
ubar_h.name = "ubar"
pbar_h = fem.Function(Qbar)
pbar_h.name = "pbar"

u_file = io.VTXWriter(msh.comm, "u.bp", [u_n._cpp_object])
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
