from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from utils import reorder_mesh, norm_L2


comm = MPI.COMM_WORLD
rank = comm.rank
out_str = f"rank {rank}:\n"

n = 8
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
# out_str += f"facets = {facets}\n"

# NOTE Despite all facets being present in the submesh, the entity map isn't
# necessarily the identity in parallel
facet_mesh, entity_map = mesh.create_submesh(msh, fdim, facets)[0:2]
# out_str += f"entity_map = {entity_map}\n"

k = 2
V = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k))
Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k - 1))
Vbar = fem.VectorFunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))
Qbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)
ubar = ufl.TrialFunction(Vbar)
vbar = ufl.TestFunction(Vbar)
pbar = ufl.TrialFunction(Qbar)
qbar = ufl.TestFunction(Qbar)

# FIXME Use CellDiameter
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
gamma = 10.0 * k**2 / h

facet_integration_entities = {1: []}
for cell in range(msh.topology.index_map(tdim).size_local):
    for local_facet in range(num_cell_facets):
        facet_integration_entities[1].extend([cell, local_facet])
# out_str += f"facet_integration_entities = {facet_integration_entities}\n"

dx_c = ufl.Measure("dx", domain=msh)
ds_c = ufl.Measure("ds", subdomain_data=facet_integration_entities, domain=msh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

inv_entity_map = [entity_map.index(entity) for entity in facets]
entity_maps = {facet_mesh: inv_entity_map}
# # out_str += f"entity_maps = {entity_maps}\n"

x = ufl.SpatialCoordinate(msh)
u_e = ufl.as_vector((x[0]**2 * (1 - x[0])**2 * (2 * x[1] - 6 * x[1]**2 + 4 * x[1]**3),
                     - x[1]**2 * (1 - x[1])**2 * (2 * x[0] - 6 * x[0]**2 + 4 * x[0]**3)))
p_e = x[0] * (1 - x[0])
f = - div(grad(u_e)) + grad(p_e)

a_00 = fem.form(inner(grad(u), grad(v)) * dx_c + gamma * inner(u, v) * ds_c(1)
                - (inner(u, dot(grad(v), n)) + inner(v, dot(grad(u), n))) * ds_c(1))
a_01 = fem.form(- inner(p, div(v)) * dx_c)
a_02 = fem.form(inner(ubar, dot(grad(v), n)) * ds_c(1) - gamma * inner(ubar, v) * ds_c(1),
                entity_maps=entity_maps)
a_03 = fem.form(inner(dot(v, n), pbar) * ds_c(1), entity_maps=entity_maps)
a_10 = fem.form(- inner(q, div(u)) * dx_c)
# Only needed to apply BC on pressure
a_11 = fem.form(fem.Constant(msh, 0.0) * inner(p, q) * dx_c)
a_20 = fem.form(inner(vbar, dot(grad(u), n)) * ds_c(1) - gamma * inner(vbar, u) * ds_c(1),
                entity_maps=entity_maps)
a_30 = fem.form(inner(dot(u, n), qbar) * ds_c(1), entity_maps=entity_maps)
a_22 = fem.form(gamma * inner(ubar, vbar) * ds_c(1), entity_maps=entity_maps)

L_0 = fem.form(inner(f, v) * dx_c)
L_1 = fem.form(inner(fem.Constant(msh, 0.0), q) * dx_c)
L_2 = fem.form(inner(ufl.as_vector((1e-9, 1e-9)), vbar) * dx_f)
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
facet_mesh_boundary_facets = [inv_entity_map[facet]
                              for facet in msh_boundary_facets]
# out_str += f"facet_mesh_boundary_facets = {facet_mesh_boundary_facets}\n"
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
bc_ubar = fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), dofs, Vbar)

# Pressure boundary condition
# TODO Locate on facet space or cell space?
# FIXME Make so it doesn't depend on diagonal direction
pressure_dof = fem.locate_dofs_geometrical(
    Q, lambda x: np.logical_and(np.isclose(x[0], 1.0),
                                np.isclose(x[1], 0.0)))
bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), pressure_dof, Q)

bcs = [bc_ubar, bc_p]

A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

out_str += f"A.norm() = {A.norm()}\n"
out_str += f"b.norm() = {b.norm()}\n"

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

out_str += f"x.norm() = {x.norm()}\n"

u = fem.Function(V)
p = fem.Function(Q)
ubar = fem.Function(Vbar)
pbar = fem.Function(Qbar)

u_offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
p_offset = u_offset + Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
u.x.array[:u_offset] = x.array_r[:u_offset]
u.x.scatter_forward()
p.x.array[:p_offset - u_offset] = x.array_r[u_offset:p_offset]
p.x.scatter_forward()

with io.VTXWriter(msh.comm, "u.bp", u) as f:
    f.write(0.0)

with io.VTXWriter(msh.comm, "p.bp", p) as f:
    f.write(0.0)

e_L2 = norm_L2(msh.comm, u - u_e)
out_str += f"e_L2 = {e_L2}\n"

if rank == 0:
    print(out_str)
