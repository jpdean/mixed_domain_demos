import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner
from mpi4py import MPI
from petsc4py import PETSc

n = 16
msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
edim = msh.topology.dim - 1
num_facets = msh.topology.create_entities(edim)
entities = mesh.locate_entities_boundary(
    msh, edim,
    lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 1.0),
                                          np.isclose(x[0], 0.0)),
                            np.logical_or(np.isclose(x[1], 1.0),
                                          np.isclose(x[1], 0.0))))
submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(
    msh, edim, entities)

V = fem.FunctionSpace(msh, ("Lagrange", 1))
W = fem.FunctionSpace(submesh, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

ds = ufl.Measure("ds", domain=msh)

f = fem.Function(V)
f.interpolate(lambda x: np.ones_like(x[0]))

g = fem.Function(W)
g.interpolate(lambda x: x[0])

with io.XDMFFile(submesh.comm, "g.xdmf", "w") as file:
    file.write_mesh(submesh)
    file.write_function(g)

msh_to_submesh = [entity_map.index(entity) if entity in entity_map else -1
                  for entity in range(num_facets)]
entity_maps = {submesh: msh_to_submesh}

a = fem.form(inner(u, v) * ufl.dx + inner(grad(u), grad(v)) * ufl.dx)
L = fem.form(inner(f, v) * ufl.dx + inner(g, v) * ds, entity_maps=entity_maps)

A = fem.petsc.assemble_matrix(a)
A.assemble()

b = fem.petsc.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

u = fem.Function(V)
ksp.solve(b, u.vector)

with io.XDMFFile(msh.comm, "poisson_neumann.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u)
