# This demo shows how Neumann boundary conditions can be interpolated
# into function spaces defined only over the Neumann boundary. This
# provides a more natural representation of boundary conditions and is
# more computationally efficient.


import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, div, dot
from mpi4py import MPI
from petsc4py import PETSc
from utils import norm_L2, markers_to_meshtags
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting


def u_e_expr(x, module=np):
    "Expression for the exact solution"
    return module.sin(module.pi * x[0]) * module.sin(module.pi * x[1])


def f_expr(x):
    "Source term"
    return 2 * np.pi**2 * u_e_expr(x)


def g_expr(x):
    "Neumann boundary condition"
    return np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])


# Create a mesh and meshtags for the Dirichlet and Neumann boundaries
n = 8
msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)

tdim = msh.topology.dim
fdim = tdim - 1
boundaries = {"dirichlet": 1, "neumann": 2}  # Tags for the boundaries
markers = [
    lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
    lambda x: np.isclose(x[0], 1.0),
]
ft = markers_to_meshtags(msh, boundaries.values(), markers, fdim)

# Create a submesh of the Neumann boundary
neumann_boundary_facets = ft.find(boundaries["neumann"])
submesh, submesh_to_mesh = mesh.create_submesh(msh, fdim, neumann_boundary_facets)[:2]

# Create function spaces
k = 3  # Polynomial degree
V = fem.functionspace(msh, ("Lagrange", k))
# Function space for the Neumann boundary condition. Note that this is defined
# only over the Neumann boundary
W = fem.functionspace(submesh, ("Lagrange", k))  
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Create integration measure and entity maps
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
# We take msh to be the integration domain, so we must provide a map from
# facets in msh to cells in submesh. This is simply the "inverse" of
# submesh_to_mesh
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
msh_to_submesh = np.full(num_facets, -1)
msh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
entity_maps = {submesh: msh_to_submesh}

# Interpolate the source term and Neumann boundary condition
f = fem.Function(V)
f.interpolate(f_expr)

g = fem.Function(W)
g.interpolate(g_expr)

# Let's write g to file to visualise it
with io.VTXWriter(msh.comm, "g.bp", g, "BP4") as file:
    file.write(0.0)

# Define forms. Since the Neumann boundary term involves funcriotns defined over
# different meshes, we must provide entity maps
a = fem.form(inner(grad(u), grad(v)) * ufl.dx)
L = fem.form(
    inner(f, v) * ufl.dx + inner(g, v) * ds(boundaries["neumann"]), entity_maps=entity_maps
)

# Dirichlet boundary condition
dirichlet_facets = ft.find(boundaries["dirichlet"])
dirichlet_dofs = fem.locate_dofs_topological(V, fdim, dirichlet_facets)
u_d = fem.Function(V)
u_d.interpolate(u_e_expr)
bc = fem.dirichletbc(u_d, dirichlet_dofs)

# Assemble matrix and vector
A = assemble_matrix(a, bcs=[bc])
A.assemble()
b = assemble_vector(L)
apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bc.set(b.array_w)

# Create solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Solve
u = fem.Function(V)
ksp.solve(b, u.x.petsc_vec)
u.x.scatter_forward()

# Write to file
with io.VTXWriter(msh.comm, "u.bp", u, "BP4") as file:
    file.write(0.0)

# Compute the error
x = ufl.SpatialCoordinate(msh)
e_L2 = norm_L2(msh.comm, u - u_e_expr(x, module=ufl))

if msh.comm.rank == 0:
    print(f"e_L2 = {e_L2}")
