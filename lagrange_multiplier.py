import numpy as np
import ufl
from dolfinx import fem, io, mesh, graph
from ufl import grad, inner
from mpi4py import MPI
from petsc4py import PETSc
import random


def create_random_mesh(corners, n, ghost_mode):
    """Create a rectangular mesh made of randomly ordered simplices"""
    if MPI.COMM_WORLD.rank == 0:
        h_x = (corners[1][0] - corners[0][0]) / n[0]
        h_y = (corners[1][1] - corners[0][1]) / n[1]

        points = [(i * h_x, j * h_y)
                  for i in range(n[0] + 1) for j in range(n[1] + 1)]

        random.seed(6)

        cells = []
        for i in range(n[0]):
            for j in range(n[1]):
                v = (n[1] + 1) * i + j
                cell_0 = [v, v + 1, v + n[1] + 2]
                random.shuffle(cell_0)
                cells.append(cell_0)

                cell_1 = [v, v + n[1] + 1, v + n[1] + 2]
                random.shuffle(cell_1)
                cells.append(cell_1)
        cells = np.array(cells)
        points = np.array(points)
    else:
        cells, points = np.empty([0, 3]), np.empty([0, 2])

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    partitioner = mesh.create_cell_partitioner(ghost_mode)
    return mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain,
                            partitioner=partitioner)


def norm_L2(comm, v):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(inner(v, v) * ufl.dx)), op=MPI.SUM))


# NOTE n must be even
n = 16
k = 1
# msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
msh = create_random_mesh(((0.0, 0.0), (1.0, 1.0)), (n, n),
                         mesh.GhostMode.shared_facet)
# msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)

tdim = msh.topology.dim
facet_dim = tdim - 1

with io.XDMFFile(msh.comm, "msh.xdmf", "w") as file:
    file.write_mesh(msh)

V = fem.FunctionSpace(msh, ("Lagrange", k))

dirichlet_facets = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                            np.isclose(x[0], 1.0)))
dirichlet_dofs = fem.locate_dofs_topological(V, facet_dim, dirichlet_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dirichlet_dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# FIXME Need to use this clumsy method until we have better support for one
# sided integrals
left_cells = mesh.locate_entities(
    msh, tdim, lambda x: x[0] <= 0.5)
submesh_lc, entity_map_lc, vertex_map_lc, geom_map_lc = mesh.create_submesh(
    msh, tdim, left_cells)
with io.XDMFFile(submesh_lc.comm, "submesh_lc.xdmf", "w") as file:
    file.write_mesh(submesh_lc)

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

# TODO Rename
submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(
    submesh_lc, facet_dim, submesh_lc_right_facets)
with io.XDMFFile(submesh.comm, "submesh.xdmf", "w") as file:
    file.write_mesh(submesh)

mp = [entity_map.index(entity) if entity in entity_map else -1
      for entity in range(sm_lc_num_facets)]

entity_maps = {msh: entity_map_lc,
               submesh: mp}
# END OF CLUMSY METHOD

W = fem.FunctionSpace(submesh, ("Lagrange", k))

lmbda = ufl.TrialFunction(W)
eta = ufl.TestFunction(W)

a_00 = fem.form(inner(grad(u), grad(v)) * ufl.dx)
a_01 = fem.form(inner(lmbda, v) * ds(1), entity_maps=entity_maps)
a_10 = fem.form(inner(u, eta) * ds(1), entity_maps=entity_maps)
f = fem.Constant(msh, PETSc.ScalarType(2.0))
L_0 = fem.form(inner(f, v) * ufl.dx)
c = fem.Constant(submesh, PETSc.ScalarType(0.25))
L_1 = fem.form(inner(c, eta) * ufl.dx)
# x = ufl.SpatialCoordinate(submesh)
# L_1 = fem.form(inner(- 0.1 * ufl.sin(ufl.pi * x[1]), eta) * ufl.dx)

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

with io.VTXWriter(msh.comm, "poisson_lm_u.bp", u) as f:
    f.write(0.0)

with io.VTXWriter(submesh.comm, "poisson_lm_lmbda.bp", lmbda) as f:
    f.write(0.0)

x = ufl.SpatialCoordinate(msh)
u_e = x[0] * (1 - x[0])

e_L2 = norm_L2(msh.comm, u - u_e)

rank = msh.comm.Get_rank()
if rank == 0:
    print(f"e_L2 = {e_L2}")
