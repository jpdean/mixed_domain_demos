# In this demo, we implement a domain decomposition scheme for
# the Poisson equation based on Nitche's method. The scheme can
# be found in "Mortaring by a method of J. A. Nitsche" by Rolf
# Stenberg. See also "A finite element method for domain
# decomposition with non-matching grids" by Becker et al.

from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, avg, div
import numpy as np
from petsc4py import PETSc
from utils import norm_L2, convert_facet_tags
import gmsh

comm = MPI.COMM_WORLD
gdim = 2

omega_0 = 1
omega_1 = 2
boundary = 3
interface = 4

gmsh.initialize()
if comm.rank == 0:
    gmsh.model.add("square_with_circle")

    factory = gmsh.model.geo

    h = 0.05

    square_points = [
        factory.addPoint(0.0, 0.0, 0.0, h),
        factory.addPoint(1.0, 0.0, 0.0, h),
        factory.addPoint(1.0, 1.0, 0.0, h),
        factory.addPoint(0.0, 1.0, 0.0, h)
    ]

    c = 0.5
    r = 0.25
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

    # gmsh.fltk.run()

partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
msh, ct, ft = io.gmshio.model_to_mesh(
    gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
gmsh.finalize()

# Create submeshes
tdim = msh.topology.dim
submesh_0, entity_map_0 = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == omega_0])[:2]
submesh_1, entity_map_1 = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == omega_1])[:2]

msh_cell_imap = msh.topology.index_map(tdim)
dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)

# Define function spaces on each submesh
V_0 = fem.FunctionSpace(submesh_0, ("Lagrange", 1))
V_1 = fem.FunctionSpace(submesh_1, ("Lagrange", 3))

# Test and trial functions
u_0 = ufl.TrialFunction(V_0)
u_1 = ufl.TrialFunction(V_1)
v_0 = ufl.TestFunction(V_0)
v_1 = ufl.TestFunction(V_1)

# Create entity maps
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
inv_entity_map_0 = np.full(num_cells, -1)
inv_entity_map_0[entity_map_0] = np.arange(len(entity_map_0))
inv_entity_map_1 = np.full(num_cells, -1)
inv_entity_map_1[entity_map_1] = np.arange(len(entity_map_1))

entity_maps = {submesh_0: inv_entity_map_0,
               submesh_1: inv_entity_map_1}

# Create measure for integration. Assign the first (cell, local facet)
# pair to the cell in omega_0, corresponding to the "+" restriction. Assign
# the second pair to the omega_1 cell, corresponding to the "-" restriction.
facet_integration_entities = []
fdim = tdim - 1
facet_imap = msh.topology.index_map(fdim)
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
domain_0_cells = ct.indices[ct.values == omega_0]
domain_1_cells = ct.indices[ct.values == omega_1]
for facet in ft.indices[ft.values == interface]:
    # Check if this facet is owned
    if facet < facet_imap.size_local:
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        cell_plus = cells[0] if cells[0] in domain_0_cells else cells[1]
        cell_minus = cells[0] if cells[0] in domain_1_cells else cells[1]
        assert cell_plus in domain_0_cells
        assert cell_minus in domain_1_cells

        # FIXME Don't use tolist
        local_facet_plus = c_to_f.links(
            cell_plus).tolist().index(facet)
        local_facet_minus = c_to_f.links(
            cell_minus).tolist().index(facet)
        facet_integration_entities.extend(
            [cell_plus, local_facet_plus, cell_minus, local_facet_minus])

        # HACK cell_minus does not exist in the left submesh, so it will
        # be mapped to index -1. This is problematic for the assembler,
        # which assumes it is possible to get the full macro dofmap for the
        # trial and test functions, despite the restriction meaning we
        # don't need the non-existant dofs. To fix this, we just map
        # cell_minus to the cell corresponding to cell plus. This will
        # just add zeros to the assembled system, since there are no
        # u("-") terms. Could map this to any cell in the submesh, but
        # I think using the cell on the other side of the facet means a
        # facet space coefficient could be used
        entity_maps[submesh_0][cell_minus] = \
            entity_maps[submesh_0][cell_plus]
        # Same hack for the right submesh
        entity_maps[submesh_1][cell_plus] = \
            entity_maps[submesh_1][cell_minus]
dS = ufl.Measure("dS", domain=msh,
                 subdomain_data=[(interface, facet_integration_entities)])

# TODO Add k dependency
gamma = 10
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

x = ufl.SpatialCoordinate(msh)
c = 1.0 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

domain_0 = "+"
domain_1 = "-"

a_00 = inner(c * grad(u_0), grad(v_0)) * dx(omega_0) \
    + gamma / avg(h) * inner(c * u_0(domain_0),
                             v_0(domain_0)) * dS(interface) \
    - inner(c * 1 / 2 * dot(grad(u_0(domain_0)), n(domain_0)),
            v_0(domain_0)) * dS(interface) \
    - inner(c * 1 / 2 * dot(grad(v_0(domain_0)), n(domain_0)),
            u_0(domain_0)) * dS(interface)

a_01 = - gamma / avg(h) * inner(c * u_1(domain_1),
                                v_0(domain_0)) * dS(interface) \
    + inner(c * 1 / 2 * dot(grad(u_1(domain_1)), n(domain_1)),
            v_0(domain_0)) * dS(interface) \
    + inner(c * 1 / 2 * dot(grad(v_0(domain_0)), n(domain_0)),
            u_1(domain_1)) * dS(interface)

a_10 = - gamma / avg(h) * inner(c * u_0(domain_0),
                                v_1(domain_1)) * dS(interface) \
    + inner(c * 1 / 2 * dot(grad(u_0(domain_0)), n(domain_0)),
            v_1(domain_1)) * dS(interface) \
    + inner(c * 1 / 2 * dot(grad(v_1(domain_1)), n(domain_1)),
            u_0(domain_0)) * dS(interface)

a_11 = inner(c * grad(u_1), grad(v_1)) * dx(omega_1) \
    + gamma / avg(h) * inner(c * u_1(domain_1),
                             v_1(domain_1)) * dS(interface) \
    - inner(c * 1 / 2 * dot(grad(u_1(domain_1)), n(domain_1)),
            v_1(domain_1)) * dS(interface) \
    - inner(c * 1 / 2 * dot(grad(v_1(domain_1)), n(domain_1)),
            u_1(domain_1)) * dS(interface)

a_00 = fem.form(a_00, entity_maps=entity_maps)
a_01 = fem.form(a_01, entity_maps=entity_maps)
a_10 = fem.form(a_10, entity_maps=entity_maps)
a_11 = fem.form(a_11, entity_maps=entity_maps)

a = [[a_00, a_01],
     [a_10, a_11]]


def u_e(x, module=np):
    return module.exp(- ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / (2 * 0.05**2)) \
        + x[0]


f = - div(c * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl)))

L_0 = inner(f, v_0) * dx(omega_0)
L_1 = inner(f, v_1) * dx(omega_1)

L_0 = fem.form(L_0, entity_maps=entity_maps)
L_1 = fem.form(L_1, entity_maps=entity_maps)

L = [L_0, L_1]

submesh_0_ft = convert_facet_tags(msh, submesh_0, entity_map_0, ft)
bound_facet_sm_0 = submesh_0_ft.indices[
    submesh_0_ft.values == boundary]

bound_dofs = fem.locate_dofs_topological(V_0, fdim, bound_facet_sm_0)

u_bc_0 = fem.Function(V_0)
u_bc_0.interpolate(u_e)
bc_0 = fem.dirichletbc(u_bc_0, bound_dofs)

bcs = [bc_0]

A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()

b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

x = A.createVecRight()

# Compute solution
ksp.solve(b, x)

u_0 = fem.Function(V_0)
u_1 = fem.Function(V_1)

offset = V_0.dofmap.index_map.size_local * V_0.dofmap.index_map_bs
u_0.x.array[:offset] = x.array_r[:offset]
u_1.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
u_0.x.scatter_forward()
u_1.x.scatter_forward()

with io.VTXWriter(msh.comm, "u_0.bp", u_0) as f:
    f.write(0.0)

with io.VTXWriter(msh.comm, "u_1.bp", u_1) as f:
    f.write(0.0)

e_L2_0 = norm_L2(msh.comm, u_0 - u_e(
    ufl.SpatialCoordinate(submesh_0), module=ufl))
e_L2_1 = norm_L2(msh.comm, u_1 - u_e(
    ufl.SpatialCoordinate(submesh_1), module=ufl))
e_L2 = np.sqrt(e_L2_0**2 + e_L2_1**2)

if msh.comm.rank == 0:
    print(e_L2)
