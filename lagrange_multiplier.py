import numpy as np
import ufl
from dolfinx import fem, io, mesh, graph
from ufl import grad, inner
from mpi4py import MPI
from petsc4py import PETSc


def norm_L2(comm, v):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(inner(v, v) * ufl.dx)), op=MPI.SUM))


# NOTE n must be even
n = 32
msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
tdim = msh.topology.dim
facet_dim = tdim - 1

entities = mesh.locate_entities(
    msh, facet_dim, lambda x: np.isclose(x[0], 0.5))
submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(
    msh, facet_dim, entities)

with io.XDMFFile(msh.comm, "msh.xdmf", "w") as file:
    file.write_mesh(msh)

with io.XDMFFile(submesh.comm, "submesh.xdmf", "w") as file:
    file.write_mesh(submesh)

V = fem.FunctionSpace(msh, ("Lagrange", 1))
W = fem.FunctionSpace(submesh, ("Lagrange", 1))

dirichlet_facets = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                            np.isclose(x[0], 1.0)))
dirichlet_dofs = fem.locate_dofs_topological(V, facet_dim, dirichlet_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dirichlet_dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

lmbda = ufl.TrialFunction(W)
eta = ufl.TestFunction(W)

# FIXME Need to use this clumsy method until we have better support for one
# sided integrals
left_cells = mesh.locate_entities(
    msh, tdim, lambda x: x[0] <= 0.5)
submesh_lc, entity_map_lc, vertex_map_lc, geom_map_lc = mesh.create_submesh(
    msh, tdim, left_cells)
with io.XDMFFile(submesh_lc.comm, "submesh_lc.xdmf", "w") as file:
    file.write_mesh(submesh_lc)

num_facets = submesh_lc.topology.create_entities(facet_dim)
submesh_lc.topology.create_connectivity(tdim - 1, 0)
sm_lc_f_to_v = submesh_lc.topology.connectivity(tdim - 1, 0)
sm_lc_num_facets = sm_lc_f_to_v.num_nodes
sm_lc_facets = graph.create_adjacencylist([sm_lc_f_to_v.links(f)
                                           for f in range(sm_lc_num_facets)])
sm_lc_facet_values = np.zeros((sm_lc_num_facets), dtype=np.int32)
submesh_lc_right_facets = mesh.locate_entities_boundary(
    submesh_lc, facet_dim, lambda x: np.isclose(x[0], 0.5))
sm_lc_facet_values[submesh_lc_right_facets] = 1
sm_lc_facet_mt = mesh.meshtags_from_entities(
    submesh_lc, facet_dim, sm_lc_facets, sm_lc_facet_values)

ds = ufl.Measure("ds", domain=submesh_lc, subdomain_data=sm_lc_facet_mt)

# mp = [entity_map.index(entity) if entity in entity_map else -1
#       for entity in range(num_facets)]
mp = []
msh_c_to_f = msh.topology.connectivity(tdim, facet_dim)
sm_lc_c_to_f = submesh_lc.topology.connectivity(tdim, facet_dim)
sm_lc_f_to_c = submesh_lc.topology.connectivity(facet_dim, tdim)
for f in range(num_facets):
    c = sm_lc_f_to_c.links(f)
    if len(c) == 1:
        cell_facets = sm_lc_c_to_f.links(c)
        local_f = np.where(cell_facets == f)[0][0]
        c_m = entity_map_lc[c[0]]
        f_msh = msh_c_to_f.links(c_m)[local_f]
        if f_msh in entity_map:
            mp.append(entity_map.index(f_msh))
        else:
            mp.append(-1)
    else:
        mp.append(-1)

entity_maps = {msh: entity_map_lc,
               submesh: mp}
# END OF CLUMSY METHOD

a_00 = fem.form(inner(grad(u), grad(v)) * ufl.dx)
a_01 = fem.form(inner(lmbda, v) * ds(1), entity_maps=entity_maps)
a_10 = fem.form(inner(u, eta) * ds(1), entity_maps=entity_maps)
f = fem.Constant(msh, PETSc.ScalarType(2.0))
L_0 = fem.form(inner(f, v) * ufl.dx)
c = fem.Constant(submesh, PETSc.ScalarType(0.25))
L_1 = fem.form(inner(c, eta) * ufl.dx)

a = [[a_00, a_01],
     [a_10, None]]
L = [L_0, L_1]

A = fem.petsc.assemble_matrix_block(a, bcs=[bc])
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=[bc])

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecLeft()
ksp.solve(b, x)

u, lmbda = fem.Function(V), fem.Function(W)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
u.x.scatter_forward()
lmbda.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
lmbda.x.scatter_forward()

with io.XDMFFile(msh.comm, "poisson_lm_u.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u)

with io.XDMFFile(submesh.comm, "poisson_lm_lmbda.xdmf", "w") as file:
    file.write_mesh(submesh)
    file.write_function(lmbda)

x = ufl.SpatialCoordinate(msh)
u_e = x[0] * (1 - x[0])

e_L2 = norm_L2(msh.comm, u - u_e)
print(e_L2)
