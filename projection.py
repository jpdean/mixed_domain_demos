# This demo shows how the trace of a function can be projected onto a
# function space defined over the boundary of a mesh


from dolfinx import mesh, fem, io
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

# Create a mesh
comm = MPI.COMM_WORLD
n = 8
msh = mesh.create_unit_square(comm, n, n)

# Create a function space for the mesh function and interpolate
V = fem.functionspace(msh, ("Lagrange", 1))
u = fem.Function(V)
u.interpolate(lambda x: np.sin(2 * np.pi * x[0]))

# Create a sub-mesh of the boundary
tdim = msh.topology.dim
fdim = tdim - 1
facets = mesh.locate_entities_boundary(
    msh,
    fdim,
    lambda x: np.isclose(x[0], 0.0)
    | np.isclose(x[0], 1.0)
    | np.isclose(x[1], 0.0)
    | np.isclose(x[1], 1.0),
)
submsh, sm_emap = mesh.create_submesh(msh, fdim, facets)[:2]

# We take msh to be the integration domain (we will pass this mess as the domain
# when creating the measure). We need to provide entity maps relating entities in
# `msh` to each other mesh in the form (here just `submsh`)
entity_maps = [sm_emap]

# Create function space on the boundary
Vbar = fem.functionspace(submsh, ("Lagrange", 1))
ubar, vbar = ufl.TrialFunction(Vbar), ufl.TestFunction(Vbar)

# Define forms for the projection
ds = ufl.Measure("ds", domain=msh)
a = fem.form(ufl.inner(ubar, vbar) * ds, entity_maps=entity_maps)
L = fem.form(ufl.inner(u, vbar) * ds, entity_maps=entity_maps)

# Assemble matrix and vector
A = assemble_matrix(a)
A.assemble()
b = assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Setup solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")

# Compute projection
ubar = fem.Function(Vbar)
ksp.solve(b, ubar.x.petsc_vec)
ubar.x.scatter_forward()

# Compute error and check it's zero to machine precision
e = u - ubar
e_L2 = np.sqrt(
    msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(e, e) * ds, entity_maps=entity_maps)))
)
assert np.isclose(e_L2, 0.0)

# Write to file
with io.VTXWriter(msh.comm, "ubar.bp", ubar, "BP4") as f:
    f.write(0.0)
