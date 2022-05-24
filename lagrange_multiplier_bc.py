# TODO Check solution is correct

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner
from mpi4py import MPI
from petsc4py import PETSc


l_x = 2.0
l_y = 1.0
n_x = 32
n_y = 16
msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (l_x, l_y)), n=(n_x, n_y),
                            cell_type=mesh.CellType.triangle,)
edim = msh.topology.dim - 1
num_facets = msh.topology.create_entities(edim)
entities = mesh.locate_entities_boundary(
    msh, edim,
    lambda x: np.logical_or(np.logical_or(np.isclose(x[0], l_x),
                                          np.isclose(x[0], 0.0)),
                            np.logical_or(np.isclose(x[1], l_y),
                                          np.isclose(x[1], 0.0))))
submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(
    msh, edim, entities)

V = fem.FunctionSpace(msh, ("Lagrange", 1))
# TODO Should this be discontinuous Lagrange?
W = fem.FunctionSpace(submesh, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
lmbda = ufl.TrialFunction(W)
mu = ufl.TestFunction(W)

# NOTE: Probably don't need to define dx
dx = ufl.Measure("dx", domain=msh)
ds = ufl.Measure("ds", domain=msh)

x = ufl.SpatialCoordinate(msh)
f = 50 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(ufl.pi * x[0])

mp = [entity_map.index(entity) if entity in entity_map else -1
      for entity in range(num_facets)]
entity_maps = {submesh: mp}

a_00 = fem.form(inner(u, v) * dx + inner(grad(u), grad(v)) * dx)
a_01 = fem.form(- inner(lmbda, v) * ds, entity_maps=entity_maps)
a_10 = fem.form(- inner(u, mu) * ds, entity_maps=entity_maps)
a_11 = None

L_0 = fem.form(inner(f, v) * dx)
L_1 = fem.form(- inner(g, mu) * ds, entity_maps=entity_maps)

a = [[a_00, a_01],
     [a_10, a_11]]
L = [L_0, L_1]

A = fem.petsc.assemble_matrix_block(a)
A.assemble()
b = fem.petsc.assemble_vector_block(L, a)

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

with io.XDMFFile(msh.comm, "poisson_u.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u)

with io.XDMFFile(submesh.comm, "poisson_lmbda.xdmf", "w") as file:
    file.write_mesh(submesh)
    file.write_function(lmbda)
