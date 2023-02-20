# NOTE: See problem in
# https://www.dealii.org/current/doxygen/deal.II/step_60.html
# and also note the comments on precondition (the Schur
# complement behaves like a Neumann-to-Dirichlet map)

import gmsh
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, div
from mpi4py import MPI
from petsc4py import PETSc
from utils import norm_L2

comm = MPI.COMM_WORLD
gdim = 2

omega_0 = 0
omega_1 = 1
boundary = 2
interface = 3

gmsh.initialize()
if comm.rank == 0:
    h = 0.05
    if gdim == 2:
        gmsh.model.add("square_with_fenics_logo")

        factory = gmsh.model.geo

        square_points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 1.0, 0.0, h),
            factory.addPoint(0.0, 1.0, 0.0, h)
        ]

        # Points for the first closed loop in the FEniCS logo
        logo_points_0 = [
            factory.addPoint(0.6017391304347826, 0.7981132075471697, 0.0, h),
            factory.addPoint(0.5782608695652174, 0.7584905660377359, 0.0, h),
            factory.addPoint(0.56, 0.730188679245283, 0.0, h),
            factory.addPoint(0.5417391304347826, 0.6924528301886792, 0.0, h),
            factory.addPoint(0.5417391304347826, 0.6584905660377358, 0.0, h),
            factory.addPoint(0.5730434782608695, 0.6320754716981132, 0.0, h),
            factory.addPoint(0.6252173913043478, 0.5962264150943397, 0.0, h),
            factory.addPoint(0.6643478260869564, 0.5641509433962264, 0.0, h),
            factory.addPoint(0.7008695652173913, 0.5320754716981132, 0.0, h),
            factory.addPoint(0.7373913043478262, 0.5037735849056604, 0.0, h),
            factory.addPoint(0.7660869565217392, 0.469811320754717, 0.0, h),
            factory.addPoint(0.7895652173913044, 0.4377358490566038, 0.0, h),
            factory.addPoint(0.8, 0.4037735849056604, 0.0, h),
            factory.addPoint(0.8052173913043479, 0.369811320754717, 0.0, h),
            factory.addPoint(0.7843478260869565, 0.34150943396226413, 0.0, h),
            factory.addPoint(0.7530434782608695, 0.3150943396226415, 0.0, h),
            factory.addPoint(0.708695652173913, 0.29245283018867924, 0.0, h),
            factory.addPoint(0.6434782608695652, 0.26037735849056604, 0.0, h),
            factory.addPoint(0.586086956521739, 0.23962264150943396, 0.0, h),
            factory.addPoint(0.5234782608695652, 0.22075471698113208, 0.0, h),
            factory.addPoint(0.4791304347826087, 0.20566037735849058, 0.0, h),
            factory.addPoint(0.4191304347826087, 0.19811320754716977, 0.0, h),
            factory.addPoint(0.46608695652173915, 0.23962264150943396, 0.0, h),
            factory.addPoint(0.5026086956521739, 0.27169811320754716, 0.0, h),
            factory.addPoint(0.5365217391304348, 0.3018867924528302, 0.0, h),
            factory.addPoint(0.557391304347826, 0.3339622641509434, 0.0, h),
            factory.addPoint(0.557391304347826, 0.36415094339622645, 0.0, h),
            factory.addPoint(0.528695652173913, 0.4018867924528302, 0.0, h),
            factory.addPoint(0.4921739130434783, 0.44716981132075473, 0.0, h),
            factory.addPoint(0.46608695652173915, 0.4811320754716981, 0.0, h),
            factory.addPoint(0.44, 0.5150943396226415, 0.0, h),
            factory.addPoint(0.4165217391304348, 0.5509433962264151, 0.0, h),
            factory.addPoint(0.39304347826086955, 0.5867924528301887, 0.0, h),
            factory.addPoint(0.3826086956521739, 0.6113207547169812, 0.0, h),
            factory.addPoint(0.3826086956521739, 0.6396226415094339, 0.0, h),
            factory.addPoint(0.4165217391304348, 0.6735849056603773, 0.0, h),
            factory.addPoint(0.45565217391304347, 0.7037735849056603, 0.0, h),
            factory.addPoint(0.5078260869565218, 0.7415094339622641, 0.0, h),
            factory.addPoint(0.5469565217391305, 0.7641509433962264, 0.0, h)
        ]

        # Points for the second closed loop in the FEniCS logo
        logo_points_1 = [
            factory.addPoint(0.30695652173913046,
                             0.5792452830188679, 0.0, h),
            factory.addPoint(0.2808695652173913,
                             0.5509433962264151, 0.0, h),
            factory.addPoint(0.24434782608695654,
                             0.5075471698113208, 0.0, h),
            factory.addPoint(0.22086956521739132,
                             0.47358490566037736, 0.0, h),
            factory.addPoint(0.20782608695652174,
                             0.41132075471698115, 0.0, h),
            factory.addPoint(0.20521739130434782,
                             0.3754716981132076, 0.0, h),
            factory.addPoint(0.23391304347826086,
                             0.3452830188679245, 0.0, h),
            factory.addPoint(0.2782608695652174,
                             0.3169811320754717, 0.0, h),
            factory.addPoint(0.3173913043478261,
                             0.2962264150943396, 0.0, h),
            factory.addPoint(0.3669565217391304,
                             0.27735849056603773, 0.0, h),
            factory.addPoint(0.3956521739130435,
                             0.2679245283018868, 0.0, h),
            factory.addPoint(0.4217391304347826,
                             0.28679245283018867, 0.0, h),
            factory.addPoint(0.44, 0.30377358490566037, 0.0, h),
            factory.addPoint(0.48434782608695653,
                             0.33962264150943394, 0.0, h),
            factory.addPoint(0.5026086956521739,
                             0.36415094339622645, 0.0, h),
            factory.addPoint(0.4973913043478261,
                             0.3905660377358491, 0.0, h),
            factory.addPoint(0.4608695652173913,
                             0.4339622641509434, 0.0, h),
            factory.addPoint(0.4295652173913044,
                             0.4660377358490566, 0.0, h),
            factory.addPoint(0.3956521739130435,
                             0.4962264150943396, 0.0, h),
            factory.addPoint(0.3669565217391304,
                             0.5245283018867924, 0.0, h),
            factory.addPoint(0.3356521739130435,
                             0.5547169811320755, 0.0, h)
        ]

        square_lines = [
            factory.addLine(square_points[0], square_points[1]),
            factory.addLine(square_points[1], square_points[2]),
            factory.addLine(square_points[2], square_points[3]),
            factory.addLine(square_points[3], square_points[0])
        ]

        # Approximate the first closed loop in the logo as two splines
        logo_lines_0 = []
        logo_lines_0.append(factory.addSpline(logo_points_0[0:22]))
        logo_lines_0.append(factory.addSpline(
            logo_points_0[21:] + [logo_points_0[0]]))

        # Approximate the second closed loop in the logo as two splines
        logo_lines_1 = []
        logo_lines_1.append(factory.addSpline(logo_points_1[0:11]))
        logo_lines_1.append(factory.addSpline(
            logo_points_1[10:] + [logo_points_1[0]]))

        # Create curves
        square_curve = factory.addCurveLoop(square_lines)
        logo_curve_0 = factory.addCurveLoop(logo_lines_0)
        logo_curve_1 = factory.addCurveLoop(logo_lines_1)

        # Create surfaces
        square_surface = factory.addPlaneSurface(
            [square_curve, logo_curve_0, logo_curve_1])
        circle_surface = factory.addPlaneSurface([logo_curve_0])
        logo_surface_1 = factory.addPlaneSurface([logo_curve_1])

        factory.synchronize()

        # Add 2D physical groups
        gmsh.model.addPhysicalGroup(2, [square_surface], omega_0)
        gmsh.model.addPhysicalGroup(
            2, [circle_surface, logo_surface_1], omega_1)

        # Add 1D physical groups
        gmsh.model.addPhysicalGroup(1, square_lines, boundary)
        gmsh.model.addPhysicalGroup(1, logo_lines_0 + logo_lines_1, interface)

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
# f = 1e-12
L_0 = fem.form(inner(f, v) * ufl.dx)

# c = fem.Constant(submesh, PETSc.ScalarType(0.25))
x_sm = ufl.SpatialCoordinate(submesh)
L_1 = fem.form(inner(u_e(x_sm), eta) * ufl.dx)
# L_1 = fem.form(inner(0.25 + 0.3 * ufl.sin(ufl.pi * x_sm[0]), eta) * ufl.dx)

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
