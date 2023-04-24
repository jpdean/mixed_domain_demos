from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from utils import norm_L2, create_random_mesh
from utils import par_print


def u_e(x):
    u_e = 1
    for i in range(tdim):
        u_e *= ufl.sin(ufl.pi * x[i])
    return u_e


comm = MPI.COMM_WORLD
rank = comm.rank

n = 16
# msh = mesh.create_unit_square(
#     comm, n, n, ghost_mode=mesh.GhostMode.none,
#     cell_type=mesh.CellType.quadrilateral)
msh = create_random_mesh(((0.0, 0.0), (1.0, 1.0)), (n, n), mesh.GhostMode.none)
# msh = mesh.create_unit_cube(
#     comm, n, n, n, ghost_mode=mesh.GhostMode.none,
#     cell_type=mesh.CellType.hexahedron)

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

k = 3
V = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
Vbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
ubar, vbar = ufl.TrialFunction(Vbar), ufl.TestFunction(Vbar)

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
gamma = 16.0 * k**2 / h

# TODO Do this with numpy
facet_integration_entities = []
for cell in range(msh.topology.index_map(tdim).size_local):
    for local_facet in range(num_cell_facets):
        facet_integration_entities.extend([cell, local_facet])

dx_c = ufl.Measure("dx", domain=msh)
ds_c = ufl.Measure("ds", subdomain_data=[
                   (1, facet_integration_entities)], domain=msh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

inv_entity_map = np.full(num_facets, -1)
inv_entity_map[entity_map] = np.arange(len(entity_map))
entity_maps = {facet_mesh: inv_entity_map}

x = ufl.SpatialCoordinate(msh)
c = 1.0 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
a_00 = fem.form(inner(c * grad(u), grad(v)) * dx_c -
                (inner(c * u, dot(grad(v), n)) * ds_c(1) +
                 inner(c * v, dot(grad(u), n)) * ds_c(1)) +
                gamma * inner(c * u, v) * ds_c(1))
a_10 = fem.form(inner(dot(grad(u), n) - gamma * u, c * vbar) * ds_c(1),
                entity_maps=entity_maps)
a_01 = fem.form(inner(dot(grad(v), n) - gamma * v, c * ubar) * ds_c(1),
                entity_maps=entity_maps)
a_11 = fem.form(gamma * inner(c * ubar, vbar) * ds_c(1),
                entity_maps=entity_maps)

f = - div(c * grad(u_e(x)))

L_0 = fem.form(inner(f, v) * dx_c)
L_1 = fem.form(inner(fem.Constant(facet_mesh, 0.0), vbar) * dx_f)

a = [[a_00, a_01],
     [a_10, a_11]]
L = [L_0, L_1]


def boundary(x):
    lr = np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    tb = np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    lrtb = lr | tb
    if tdim == 2:
        return lrtb
    else:
        assert tdim == 3
        fb = np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)
        return lrtb | fb


msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
facet_mesh_boundary_facets = inv_entity_map[msh_boundary_facets]
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, Vbar)

A = fem.petsc.assemble_matrix_block(a, bcs=[bc])
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=[bc])

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

u, ubar = fem.Function(V), fem.Function(Vbar)

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
ubar.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
u.x.scatter_forward()
ubar.x.scatter_forward()

with io.VTXWriter(msh.comm, "u.bp", u) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "ubar.bp", ubar) as f:
    f.write(0.0)

x = ufl.SpatialCoordinate(msh)
e_u = norm_L2(msh.comm, u - u_e(x))
x_bar = ufl.SpatialCoordinate(facet_mesh)
e_ubar = norm_L2(msh.comm, ubar - u_e(x_bar))
par_print(comm, f"e_u = {e_u}")
par_print(comm, f"e_ubar = {e_ubar}")
