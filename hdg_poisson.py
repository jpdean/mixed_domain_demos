from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div
import numpy as np
from petsc4py import PETSc


def norm_L2(comm, v):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(inner(v, v) * ufl.dx)), op=MPI.SUM))


comm = MPI.COMM_WORLD
rank = comm.rank
out_str = f"rank {rank}:\n"

n = 4
# msh = mesh.create_unit_square(
#     comm, n, n, ghost_mode=mesh.GhostMode.none)
msh = mesh.create_unit_cube(
    comm, n, n, n, ghost_mode=mesh.GhostMode.none)

# TODO Don't hardcode
num_cell_vertices = 4

# Currently, permutations are not working in parallel, so reorder the
# mesh
# FIXME For a high-order mesh, the geom has more dofs so need to modify this
tdim = msh.topology.dim
c_to_v = msh.topology.connectivity(tdim, 0)
geom_dofmap = msh.geometry.dofmap
vertex_imap = msh.topology.index_map(0)
geom_imap = msh.geometry.index_map()
for i in range(0, len(c_to_v.array), num_cell_vertices):
    topo_perm = np.argsort(vertex_imap.local_to_global(c_to_v.array[i:i+num_cell_vertices]))
    geom_perm = np.argsort(geom_imap.local_to_global(geom_dofmap.array[i:i+num_cell_vertices]))

    c_to_v.array[i:i+num_cell_vertices] = c_to_v.array[i:i+num_cell_vertices][topo_perm]
    geom_dofmap.array[i:i+num_cell_vertices] = geom_dofmap.array[i:i+num_cell_vertices][geom_perm]

fdim = tdim - 1
msh.topology.create_entities(fdim)

facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)
# out_str += f"facets = {facets}\n"

# TODO Figure out why entity_map isn't the identity and if this is
# an issue or not
facet_mesh, entity_map = mesh.create_submesh(msh, fdim, facets)[0:2]
# out_str += f"entity_map = {entity_map}\n"

k = 1
V = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
Vbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
ubar = ufl.TrialFunction(Vbar)
vbar = ufl.TestFunction(Vbar)

# FIXME Use CellDiameter
h = 1 / n
gamma = 10.0 * k**2 / h
n = ufl.FacetNormal(msh)

facet_integration_entities = {1: []}
for cell in range(msh.topology.index_map(tdim).size_local):
    # TODO Don't hardcode number of facets per cell
    for local_facet in range(4):
        facet_integration_entities[1].extend([cell, local_facet])
# out_str += f"facet_integration_entities = {facet_integration_entities}\n"

dx_c = ufl.Measure("dx", domain=msh)
ds_c = ufl.Measure("ds", subdomain_data=facet_integration_entities, domain=msh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

inv_entity_map = [entity_map.index(entity) for entity in facets]
entity_maps = {facet_mesh: inv_entity_map}
# out_str += f"entity_maps = {entity_maps}\n"

a_00 = fem.form(inner(grad(u), grad(v)) * dx_c -
                (inner(u, dot(grad(v), n)) * ds_c(1) +
                 inner(v, dot(grad(u), n)) * ds_c(1)) +
                gamma * inner(u, v) * ds_c(1))
a_10 = fem.form(inner(dot(grad(u), n) - gamma * u, vbar) * ds_c(1),
                entity_maps=entity_maps)
a_01 = fem.form(inner(dot(grad(v), n) - gamma * v, ubar) * ds_c(1),
                entity_maps=entity_maps)
# TODO Check below
a_11 = fem.form(gamma * inner(ubar, vbar) * ds_c(1),
                entity_maps=entity_maps)
# a_11 = fem.form(2 * gamma * inner(ubar, vbar) * dx_f)

x = ufl.SpatialCoordinate(msh)
u_e = 1
for i in range(tdim):
    u_e *= ufl.sin(ufl.pi * x[i])
f = - div(grad(u_e))

L_0 = fem.form(inner(f, v) * dx_c)
L_1 = fem.form(inner(fem.Constant(facet_mesh, 0.0), vbar) * dx_f)

a = [[a_00, a_01],
     [a_10, a_11]]
L = [L_0, L_1]


def boundary(x):
    lr = np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    tb = np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    fb = np.logical_or(np.isclose(x[2], 0.0), np.isclose(x[2], 1.0))
    return np.logical_or(np.logical_or(lr, tb), fb)


# NOTE Locating boundary facets on the mesh to ensure we don't hit
# bug caused by strange ghosting on the facet mesh
msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
facet_mesh_boundary_facets = [inv_entity_map[facet]
                              for facet in msh_boundary_facets]
# out_str += f"facet_mesh_boundary_facets = {facet_mesh_boundary_facets}\n"

dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, Vbar)

A = fem.petsc.assemble_matrix_block(a, bcs=[bc])
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=[bc])

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
ubar = fem.Function(Vbar)

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
ubar.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
u.x.scatter_forward()
ubar.x.scatter_forward()

with io.VTXWriter(msh.comm, "u.bp", u) as f:
    f.write(0.0)

# FIXME Why are there extra facets?
with io.VTXWriter(msh.comm, "ubar.bp", ubar) as f:
    f.write(0.0)

e_L2 = norm_L2(msh.comm, u - u_e)

out_str += f"e_L2 = {e_L2}\n"

print(out_str)
