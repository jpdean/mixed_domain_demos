# TODO https://www.dealii.org/current/doxygen/deal.II/step_60.html
# Create mesh of FEniCS logo

# TODO 3D

import gmsh
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, div
from mpi4py import MPI
from petsc4py import PETSc
from utils import norm_L2

comm = MPI.COMM_WORLD
gdim = 3

omega_0 = 0
omega_1 = 1
boundary = 2
interface = 3

gmsh.initialize()
if comm.rank == 0:
    h = 0.2
    if gdim == 2:
        gmsh.model.add("square_with_circle")

        factory = gmsh.model.geo

        square_points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 1.0, 0.0, h),
            factory.addPoint(0.0, 1.0, 0.0, h)
        ]

        c = 0.5
        r = 0.2
        circle_points = [
            factory.addPoint(c, c, 0.0, h),
            factory.addPoint(c + r, c, 0.0, h),
            factory.addPoint(c, c + r, 0.0, h),
            factory.addPoint(c - r, c, 0.0, h),
            factory.addPoint(c, c - r, 0.0, h)
        ]

        square_lines = [
            factory.addLine(square_points[0], square_points[1]),
            factory.addLine(square_points[1], square_points[2]),
            factory.addLine(square_points[2], square_points[3]),
            factory.addLine(square_points[3], square_points[0])
        ]

        circle_lines = [
            factory.addCircleArc(
                circle_points[1], circle_points[0], circle_points[2]),
            factory.addCircleArc(
                circle_points[2], circle_points[0], circle_points[3]),
            factory.addCircleArc(
                circle_points[3], circle_points[0], circle_points[4]),
            factory.addCircleArc(
                circle_points[4], circle_points[0], circle_points[1])
        ]

        square_curve = factory.addCurveLoop(square_lines)
        circle_curve = factory.addCurveLoop(circle_lines)

        square_surface = factory.addPlaneSurface([square_curve, circle_curve])
        circle_surface = factory.addPlaneSurface([circle_curve])

        factory.synchronize()

        gmsh.model.addPhysicalGroup(2, [square_surface], omega_0)
        gmsh.model.addPhysicalGroup(2, [circle_surface], omega_1)

        gmsh.model.addPhysicalGroup(1, square_lines, boundary)
        gmsh.model.addPhysicalGroup(1, circle_lines, interface)

        gmsh.model.mesh.generate(2)

    elif gdim == 3:
        gmsh.model.add("box_with_sphere")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        sphere = gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.25)

        ov, ovv = gmsh.model.occ.fragment([(3, box)], [(3, sphere)])

        # print("fragment produced volumes:")
        # for e in ov:
        #     print(e)

        # print("before/after fragment relations:")
        # for e in zip([(3, box), (3, sphere)], ovv):
        #     print("parent " + str(e[0]) + " -> child " + str(e[1]))

        gmsh.model.occ.synchronize()

        boundary_dim_tags = gmsh.model.getBoundary([ov[0], ov[1]])
        interface_dim_tags = gmsh.model.getBoundary([ov[0]])

        gmsh.model.addPhysicalGroup(3, [ov[0][1]], omega_0)
        gmsh.model.addPhysicalGroup(3, [ov[1][1]], omega_1)
        gmsh.model.addPhysicalGroup(
            2, [surface[1] for surface in boundary_dim_tags], boundary)
        gmsh.model.addPhysicalGroup(
            2, [surface[1] for surface in interface_dim_tags], interface)

        # Assign a mesh size to all the points:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

        gmsh.model.mesh.generate(3)

partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
msh, ct, ft = io.gmshio.model_to_mesh(
    gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
gmsh.finalize()

# with io.XDMFFile(comm, "msh.xdmf", "w") as f:
#     f.write_mesh(msh)
#     f.write_meshtags(ct)
#     f.write_meshtags(ft)

k = 3

V = fem.FunctionSpace(msh, ("Lagrange", k))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Create Dirichlet boundary condition
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_entities(fdim)
dirichlet_facets = ft.indices[ft.values == boundary]
dirichlet_dofs = fem.locate_dofs_topological(V, fdim, dirichlet_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dirichlet_dofs, V)


# Create submesh for Lagrange multiplier
interface_facets = ft.indices[ft.values == interface]
submesh, entity_map = mesh.create_submesh(msh, fdim, interface_facets)[0:2]


# Create function space for the Lagrange multiplier
W = fem.FunctionSpace(submesh, ("Lagrange", k))
lmbda = ufl.TrialFunction(W)
eta = ufl.TestFunction(W)

facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
inv_entity_map = np.full(num_facets, -1)
inv_entity_map[entity_map] = np.arange(len(entity_map))
entity_maps = {submesh: inv_entity_map}

# Create measure for integration
facet_integration_entities = {interface: []}
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
for facet in interface_facets:
    # Check if this facet is owned
    if facet < facet_imap.size_local:
        # Get a cell connected to the facet
        cell = f_to_c.links(facet)[0]
        local_facet = c_to_f.links(cell).tolist().index(facet)
        facet_integration_entities[interface].extend([cell, local_facet])
ds = ufl.Measure("ds", subdomain_data=facet_integration_entities, domain=msh)


def u_e(x):
    u_e = 1
    for i in range(tdim):
        u_e *= ufl.sin(ufl.pi * x[i])
    return u_e


# Define forms
a_00 = fem.form(inner(grad(u), grad(v)) * ufl.dx)
a_01 = fem.form(inner(lmbda, v) * ds(interface), entity_maps=entity_maps)
a_10 = fem.form(inner(u, eta) * ds(interface), entity_maps=entity_maps)

# f = fem.Constant(msh, PETSc.ScalarType(2.0))
x_msh = ufl.SpatialCoordinate(msh)
f = - div(grad(u_e(x_msh)))
L_0 = fem.form(inner(f, v) * ufl.dx)

# c = fem.Constant(submesh, PETSc.ScalarType(0.25))
x_sm = ufl.SpatialCoordinate(submesh)
L_1 = fem.form(inner(u_e(x_sm), eta) * ufl.dx)

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
with io.VTXWriter(msh.comm, "u.bp", u) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "lmbda.bp", lmbda) as f:
    f.write(0.0)

# Compute L^2-norm of error
e_L2 = norm_L2(msh.comm, u - u_e(x_msh))
rank = msh.comm.Get_rank()
if rank == 0:
    print(f"e_L2 = {e_L2}")
    # print(1 / (msh.topology.index_map(2).size_global)**(1/msh.topology.dim))
