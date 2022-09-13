import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner
from mpi4py import MPI
from petsc4py import PETSc


def boundary_marker(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)),
                         np.logical_or(np.isclose(x[1], 0.0),
                                       np.isclose(x[1], 1.0)))


n = 4
msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
fdim = msh.topology.dim - 1
num_facets = msh.topology.create_entities(fdim)
boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary_marker)
submesh, entity_map = mesh.create_submesh(msh, fdim, boundary_facets)[0:2]

V = fem.FunctionSpace(msh, ("Lagrange", 1))
W = fem.FunctionSpace(submesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

ds = ufl.Measure("ds", domain=msh)


# with io.XDMFFile(submesh.comm, "g.xdmf", "w") as file:
#     file.write_mesh(submesh)
#     file.write_function(g)

entity_maps = {submesh: [entity_map.index(entity)
                         if entity in entity_map else -1
                         for entity in range(num_facets)]}

a = fem.form(inner(u, v) * ufl.dx + inner(grad(u), grad(v)) * ufl.dx)
# FIXME Problem packing coefficients for both cell and facet space in one form
# f = fem.Function(V)
# f.interpolate(lambda x: np.ones_like(x[0]))
g = fem.Function(W)
g.interpolate(lambda x: x[0])
# L = fem.form(inner(f, v) * ufl.dx + inner(g, v) * ds, entity_maps=entity_maps)
L = fem.form(inner(g, v) * ds, entity_maps=entity_maps)

A = fem.petsc.assemble_matrix(a)
A.assemble()

b = fem.petsc.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Create solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Solve
u = fem.Function(V)
ksp.solve(b, u.vector)
u.x.scatter_forward()

# Write to file
with io.VTXWriter(msh.comm, "u.bp", u) as f:
    f.write(0.0)
