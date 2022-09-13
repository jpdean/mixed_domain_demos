# TODO Check solution is correct

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, div
from mpi4py import MPI
from petsc4py import PETSc
from utils import norm_L2


def boundary_marker(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], l_x)),
                         np.logical_or(np.isclose(x[1], 0.0),
                                       np.isclose(x[1], l_y)))


# Create mesh
l_x = 2.0
l_y = 1.0
n_x = 32
n_y = 16
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, points=((0.0, 0.0), (l_x, l_y)), n=(n_x, n_y))

# Create submesh of the boundary
tdim = msh.topology.dim
fdim = tdim - 1
num_facets = msh.topology.create_entities(fdim)
boundary_facets = mesh.locate_entities_boundary(
    msh, fdim, boundary_marker)
submesh, entity_map = mesh.create_submesh(msh, fdim, boundary_facets)[0:2]

# Create function spaces on the mesh and submesh
V = fem.FunctionSpace(msh, ("Lagrange", 1))
W = fem.FunctionSpace(submesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
lmbda = ufl.TrialFunction(W)
mu = ufl.TestFunction(W)

# Create measure for integral over boundary
ds = ufl.Measure("ds", domain=msh)

# Create manufactured solution
x = ufl.SpatialCoordinate(msh)
u_e = 1
for i in range(tdim):
    u_e *= ufl.cos(ufl.pi * x[i])
f = u_e - div(grad(u_e))

# Dirichlet boundary condition (enforced through Lagrange multiplier)
u_d = u_e

# Define entity maps and forms
entity_maps = {submesh: [entity_map.index(entity) if entity in entity_map else -1
                         for entity in range(num_facets)]}
a_00 = fem.form(inner(u, v) * ufl.dx + inner(grad(u), grad(v)) * ufl.dx)
a_01 = fem.form(- inner(lmbda, v) * ds, entity_maps=entity_maps)
a_10 = fem.form(- inner(u, mu) * ds, entity_maps=entity_maps)
a_11 = None
L_0 = fem.form(inner(f, v) * ufl.dx)
L_1 = fem.form(- inner(u_d, mu) * ds, entity_maps=entity_maps)

a = [[a_00, a_01],
     [a_10, a_11]]
L = [L_0, L_1]

# Assemble matrices
A = fem.petsc.assemble_matrix_block(a)
A.assemble()
b = fem.petsc.assemble_vector_block(L, a)

# Solve
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecLeft()
ksp.solve(b, x)

# Recover solution
u, lmbda = fem.Function(V), fem.Function(W)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
u.x.scatter_forward()
lmbda.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
lmbda.x.scatter_forward()

with io.VTXWriter(msh.comm, "u.bp", u) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "lmbda.bp", lmbda) as f:
    f.write(0.0)

e_L2 = norm_L2(msh.comm, u - u_e)

if msh.comm.rank == 0:
    print(f"e_L2 = {e_L2}")
