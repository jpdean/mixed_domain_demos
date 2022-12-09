from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from utils import norm_L2


def u_e(x):
    if type(x) == ufl.SpatialCoordinate:
        module = ufl
    else:
        module = np

    return module.sin(module.pi * x[0]) * module.cos(module.pi * x[1])


comm = MPI.COMM_WORLD

n = 8
msh = mesh.create_unit_square(comm, n, n)

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

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
ubar = ufl.TrialFunction(Vbar)
vbar = ufl.TestFunction(Vbar)

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
gamma = 16.0 * k**2 / h

# TODO Do this with numpy
all_facets = 0
facet_integration_entities = {all_facets: []}
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

kappa = fem.Constant(msh, PETSc.ScalarType(0.1))

a_00 = inner(kappa * grad(u), grad(v)) * dx_c \
    - inner(kappa * dot(grad(u), n), v) * ds_c(all_facets) \
    - inner(kappa * u, dot(grad(v), n)) * ds_c(all_facets) \
    + gamma * inner(kappa * u, v) * ds_c(all_facets)
a_01 = inner(kappa * ubar, dot(grad(v), n)) * ds_c(all_facets) \
    - gamma * inner(kappa * ubar, v) * ds_c(all_facets)
a_10 = inner(kappa * dot(grad(u), n), vbar) * ds_c(all_facets) \
    - gamma * inner(kappa * u, vbar) * ds_c(all_facets)
a_11 = gamma * inner(kappa * ubar, vbar) * ds_c(all_facets)

a_00 = fem.form(a_00)
a_01 = fem.form(a_01, entity_maps=entity_maps)
a_10 = fem.form(a_10, entity_maps=entity_maps)
a_11 = fem.form(a_11, entity_maps=entity_maps)

x = ufl.SpatialCoordinate(msh)
f = - div(kappa * grad(u_e(x)))

L_0 = fem.form(inner(f, v) * dx_c)
L_1 = fem.form(inner(fem.Constant(facet_mesh, 0.0), vbar) * dx_f)

a = [[a_00, a_01],
     [a_10, a_11]]
L = [L_0, L_1]


def boundary(x):
    lr = np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    tb = np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    return lr | tb


msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
facet_mesh_boundary_facets = inv_entity_map[msh_boundary_facets]
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
u_bc = fem.Function(Vbar)
u_bc.interpolate(u_e)
bc = fem.dirichletbc(u_bc, dofs)

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

u = fem.Function(V)
ubar = fem.Function(Vbar)

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
e_L2 = norm_L2(msh.comm, u - u_e(x))

if comm.rank == 0:
    print(f"e_L2 = {e_L2}")
