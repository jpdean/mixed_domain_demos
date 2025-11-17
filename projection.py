# This demo shows how the trace of a function can be projected onto a
# function space defined over the boundary of a mesh


from dolfinx import mesh, fem, io
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem

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
a = ufl.inner(ubar, vbar) * ds
L = ufl.inner(u, vbar) * ds

petsc_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
problem = LinearProblem(
    a,
    L,
    bcs=[],
    petsc_options_prefix="projection_",
    petsc_options=petsc_opts,
    entity_maps=entity_maps,
)
ubar = problem.solve()

# Compute error and check it's zero to machine precision
e = u - ubar
e_L2 = np.sqrt(
    msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(e, e) * ds, entity_maps=entity_maps)))
)
assert np.isclose(e_L2, 0.0)

# Write to file
with io.VTXWriter(msh.comm, "ubar.bp", ubar) as f:
    f.write(0.0)
