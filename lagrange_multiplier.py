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


n = 16
assert n % 2 == 0  # NOTE n must be even
k = 1
# msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
msh = create_random_mesh(((0.0, 0.0), (1.0, 1.0)), (n, n),
                         mesh.GhostMode.shared_facet)
# msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


with io.XDMFFile(msh.comm, "msh.xdmf", "w") as file:
    file.write_mesh(msh)

V = fem.FunctionSpace(msh, ("Lagrange", k))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Create Dirichlet boundary condition
tdim = msh.topology.dim
facet_dim = tdim - 1
dirichlet_facets = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                            np.isclose(x[0], 1.0)))
dirichlet_dofs = fem.locate_dofs_topological(V, facet_dim, dirichlet_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dirichlet_dofs, V)

# FIXME Need to use this clumsy method until we have better support for one
# sided integrals
# Create a submesh of the cells on the left side of the mesh
left_cells = mesh.locate_entities(
    msh, tdim, lambda x: x[0] <= 0.5)
submesh_left_cells, entity_map_left_cells = mesh.create_submesh(
    msh, tdim, left_cells)[0:2]
with io.XDMFFile(submesh_left_cells.comm, "submesh_left_cells.xdmf", "w") as file:
    file.write_mesh(submesh_left_cells)

# Tag the facets on the right boundary of the submesh. These correspond to
# the centre facets of the original mesh
# NOTE Numbered with respect to submesh_left_cells
centre_facets = mesh.locate_entities_boundary(
    submesh_left_cells, facet_dim, lambda x: np.isclose(x[0], 0.5))
mt = mesh.meshtags(
    submesh_left_cells, facet_dim, centre_facets, 1)

# Create an exterior facet measure on the submesh using the meshtags. Forms
# with ds(1) therefore integrates over the centre facets of the orignial mesh
ds = ufl.Measure("ds", domain=submesh_left_cells, subdomain_data=mt)

# Create a submesh of the centre facets of the mesh to define the function space
# for the Lagrange multiplier
submesh_centre_facets, entity_map_centre_facets = mesh.create_submesh(
    submesh_left_cells, facet_dim, centre_facets)[0:2]
with io.XDMFFile(submesh_centre_facets.comm,
                 "submesh_centre_facets.xdmf", "w") as file:
    file.write_mesh(submesh_centre_facets)

# We need to provide entitiy maps for both the original mesh and the submesh
# of the centre facets
facet_imap = submesh_left_cells.topology.index_map(facet_dim)
sm_lc_num_facets = facet_imap.size_local + facet_imap.num_ghosts
entity_maps = {msh: entity_map_left_cells,
               submesh_centre_facets: [entity_map_centre_facets.index(entity)
                                       if entity in entity_map_centre_facets else -1
                                       for entity in range(sm_lc_num_facets)]}
# END OF CLUMSY METHOD

# Create function space for the Lagrange multiplier
W = fem.FunctionSpace(submesh_centre_facets, ("Lagrange", k))
lmbda = ufl.TrialFunction(W)
eta = ufl.TestFunction(W)

# Define forms
a_00 = fem.form(inner(grad(u), grad(v)) * ufl.dx)
a_01 = fem.form(inner(lmbda, v) * ds(1), entity_maps=entity_maps)
a_10 = fem.form(inner(u, eta) * ds(1), entity_maps=entity_maps)
f = fem.Constant(msh, PETSc.ScalarType(2.0))
L_0 = fem.form(inner(f, v) * ufl.dx)
c = fem.Constant(submesh_centre_facets, PETSc.ScalarType(0.25))
L_1 = fem.form(inner(c, eta) * ufl.dx)
# x = ufl.SpatialCoordinate(submesh)
# L_1 = fem.form(inner(- 0.1 * ufl.sin(ufl.pi * x[1]), eta) * ufl.dx)

a = [[a_00, a_01],
     [a_10, None]]
L = [L_0, L_1]

# Use block assembly
A = fem.petsc.assemble_matrix_block(a, bcs=[bc])
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=[bc])

# Configure solver
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

# Write to file
with io.VTXWriter(msh.comm, "poisson_lm_u.bp", u) as f:
    f.write(0.0)
with io.VTXWriter(submesh_centre_facets.comm, "poisson_lm_lmbda.bp", lmbda) as f:
    f.write(0.0)

# Compute L^2-norm of error
x = ufl.SpatialCoordinate(msh)
u_e = x[0] * (1 - x[0])
e_L2 = norm_L2(msh.comm, u - u_e)
rank = msh.comm.Get_rank()
if rank == 0:
    print(f"e_L2 = {e_L2}")
