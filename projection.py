# Demo showing how the trace of a solution can be projected onto the boundary
# of a mesh
from dolfinx import mesh, fem, io
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc

# Create a unit square mesh
comm = MPI.COMM_WORLD
n = 8
msh = mesh.create_unit_square(comm, n, n)

# Create a function space for the mesh function and interpolate
V = fem.FunctionSpace(msh, ("Lagrange", 1))
u = fem.Function(V)
u.interpolate(lambda x: np.sin(2 * np.pi * x[0]))

# Create a submesh of the boundary
tdim = msh.topology.dim
facets = mesh.locate_entities_boundary(
    msh, tdim - 1, lambda x:
        np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
        np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
submsh, sm_to_msh = mesh.create_submesh(msh, tdim - 1, facets)[:2]

# Create msh to submsh entity map
num_facets = msh.topology.index_map(tdim - 1).size_local + \
    msh.topology.index_map(tdim - 1).num_ghosts
msh_to_sm = np.full(num_facets, -1)
msh_to_sm[sm_to_msh] = np.arange(len(sm_to_msh))
entity_maps = {submsh: msh_to_sm}

# Create function space on the boundary
Vbar = fem.FunctionSpace(submsh, ("Lagrange", 1))
ubar = ufl.TrialFunction(Vbar)
vbar = ufl.TestFunction(Vbar)

# Define forms for the projection
ds = ufl.Measure("ds", domain=msh)
a = fem.form(ufl.inner(ubar, vbar) * ds, entity_maps=entity_maps)
L = fem.form(ufl.inner(u, vbar) * ds, entity_maps=entity_maps)

# Assemble matrix and vector
A = fem.petsc.assemble_matrix(a)
A.assemble()
b = fem.petsc.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Setup solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")

# Compute projection
ubar = fem.Function(Vbar)
ksp.solve(b, ubar.vector)
ubar.x.scatter_forward()

# Compute error and check it's zero to machine precision
e = u - ubar
e_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(
    fem.form(ufl.inner(e, e) * ds, entity_maps=entity_maps))))
assert np.isclose(e_L2, 0.0)

# Write to file
with io.VTXWriter(msh.comm, "ubar.bp", ubar) as f:
    f.write(0.0)