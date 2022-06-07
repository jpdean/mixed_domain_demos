import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, dx
from mpi4py import MPI
from petsc4py import PETSc

n = 8
msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
facet_dim = msh.topology.dim - 1
num_facets = msh.topology.create_entities(facet_dim)

V = fem.FunctionSpace(msh, ("Lagrange", 1))

dirichlet_facets = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                            np.isclose(x[0], 1.0)))
dirichlet_dofs = fem.locate_dofs_topological(V, facet_dim, dirichlet_facets)
bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dirichlet_dofs, V=V)

sm_entities = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.isclose(x[2], 0.0))
submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(
    msh, facet_dim, sm_entities)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Function(V)
f.interpolate(lambda x: np.sin(np.pi * x[0])
              * np.sin(np.pi * x[1])
              * np.sin(np.pi * x[2]))

W = fem.FunctionSpace(submesh, ("Lagrange", 1))

u_sm = ufl.TrialFunction(W)
v_sm = ufl.TestFunction(W)

f_sm = fem.Function(W)
f_sm.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

sm_facet_dim = submesh.topology.dim - 1
sm_boundary_facets = mesh.locate_entities_boundary(
    submesh, sm_facet_dim,
    lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                          np.isclose(x[0], 1.0)),
                            np.logical_or(np.isclose(x[1], 0.0),
                                          np.isclose(x[1], 1.0))))
submesh_1, entity_map_1, vertex_map_1, geom_map_1 = mesh.create_submesh(
    submesh, sm_facet_dim, sm_boundary_facets)
X = fem.FunctionSpace(submesh_1, ("Lagrange", 1))
g = fem.Function(X)
g.interpolate(lambda x: x[0]**2)
with io.XDMFFile(submesh_1.comm, "g.xdmf", "w") as file:
    file.write_mesh(submesh_1)
    file.write_function(g)

a_sm = fem.form(inner(u_sm, v_sm) * dx + inner(grad(u_sm), grad(v_sm)) * dx)
L_sm = fem.form(inner(f_sm, v_sm) * dx)

A_sm = fem.petsc.assemble_matrix(a_sm)
A_sm.assemble()
b_sm = fem.petsc.assemble_vector(L_sm)
b_sm.ghostUpdate(addv=PETSc.InsertMode.ADD,
                 mode=PETSc.ScatterMode.REVERSE)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A_sm)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

u_sm = fem.Function(W)
ksp.solve(b_sm, u_sm.vector)
u_sm.x.scatter_forward()

with io.XDMFFile(submesh.comm, "u_sm.xdmf", "w") as file:
    file.write_mesh(submesh)
    file.write_function(u_sm)

msh_to_submesh = [entity_map.index(entity) if entity in entity_map else -1
                  for entity in range(num_facets)]
entity_maps = {submesh: msh_to_submesh}

ds = ufl.Measure("ds", domain=msh)

a = fem.form(inner(grad(u), grad(v)) * dx)
L = fem.form(inner(f, v) * dx + inner(u_sm, v) * ds,
             entity_maps=entity_maps)

A = fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()

b = fem.petsc.assemble_vector(L)
fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, [bc])

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

u = fem.Function(V)
ksp.solve(b, u.vector)

with io.XDMFFile(msh.comm, "u.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u)
