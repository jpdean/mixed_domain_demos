# Scheme based on "A finite element method for domain decomposition
# with non-matching grids" by Becker et al. but with a DG scheme to
# solve the advection diffusion equation on half of the domain, and
# a standard CG Poisson solver on the other half.

# Consider a square domain on which we wish to solve the
# advection diffusion equations. The velocity field is given by
# (0.5 - x_1, 0.0) in the bottom half of the domain, and (0.0, 0.0)
# in the top half. We solve the bottom half of the domain with a
# DG advection-diffusion solver, and the top half with a standard
# CG solver. We enforce the Dirichlet boundary condition weakly
# for the DG scheme and strongly for the CG scheme. The assumed
# solution is u = sin(\pi * x_0) * sin(\pi * x_1). In this problem,
# the bottom half can be thought of as a fluid and the top half
# a solid, and the unknown u is the temperature field.

# NOTE: Since the velocity goes to zero at the interface x[1] = 0.5,
# the coupling is due only to the diffusion. No advective interface
# terms have been added

from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, avg, div, jump
import numpy as np
from petsc4py import PETSc
from utils import norm_L2, convert_facet_tags
import gmsh


# Set some parameters
num_time_steps = 10
k_0 = 3  # Polynomial degree in omega_0
k_1 = 3  # Polynomial degree in omgea_1
delta_t = 1  # TODO Make constant

comm = MPI.COMM_WORLD

vol_ids = {"omega_0": 1,
           "omega_1": 2}
boundary_0 = 3
boundary_1 = 4
interface = 5
omega_0_int_facets = 6

gdim = 2
gmsh.initialize()
if comm.rank == 0:
    gmsh.model.add("model")
    factory = gmsh.model.geo

    h = 0.05
    points = [
        factory.addPoint(0.0, 0.0, 0.0, h),
        factory.addPoint(1.0, 0.0, 0.0, h),
        factory.addPoint(1.0, 0.5, 0.0, h),
        factory.addPoint(0.0, 0.5, 0.0, h),
        factory.addPoint(0.0, 1.0, 0.0, h),
        factory.addPoint(1.0, 1.0, 0.0, h)
    ]

    square_0_lines = [
        factory.addLine(points[0], points[1]),
        factory.addLine(points[1], points[2]),
        factory.addLine(points[2], points[3]),
        factory.addLine(points[3], points[0])
    ]

    square_1_lines = [
        square_0_lines[2],
        factory.addLine(points[3], points[4]),
        factory.addLine(points[4], points[5]),
        factory.addLine(points[5], points[2]),
    ]

    square_0_curve = factory.addCurveLoop(square_0_lines)
    square_1_curve = factory.addCurveLoop(square_1_lines)

    square_0_surface = factory.addPlaneSurface([square_0_curve])
    square_1_surface = factory.addPlaneSurface([square_1_curve])

    factory.synchronize()

    gmsh.model.addPhysicalGroup(2, [square_0_surface], vol_ids["omega_0"])
    gmsh.model.addPhysicalGroup(2, [square_1_surface], vol_ids["omega_1"])

    gmsh.model.addPhysicalGroup(1, [square_0_lines[0],
                                    square_0_lines[1],
                                    square_0_lines[3]], boundary_0)

    gmsh.model.addPhysicalGroup(1, [square_1_lines[1],
                                    square_1_lines[2],
                                    square_1_lines[3]], boundary_1)

    gmsh.model.addPhysicalGroup(1, [square_0_lines[2]], interface)

    gmsh.model.mesh.generate(2)

    # gmsh.fltk.run()

partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
msh, ct, ft = io.gmshio.model_to_mesh(
    gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
gmsh.finalize()

# Create submeshes
tdim = msh.topology.dim
submesh_0, entity_map_0 = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == vol_ids["omega_0"]])[:2]
submesh_1, entity_map_1 = mesh.create_submesh(
    msh, tdim, ct.indices[ct.values == vol_ids["omega_1"]])[:2]

msh_cell_imap = msh.topology.index_map(tdim)
dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)

# Define function spaces on each submesh
V_0 = fem.FunctionSpace(submesh_0, ("Discontinuous Lagrange", k_0))
V_1 = fem.FunctionSpace(submesh_1, ("Lagrange", k_1))

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
# facet_integration_entities = {interface: [],
#                               omega_0_int_facets: []}
interface_entities = []
fdim = tdim - 1
facet_imap = msh.topology.index_map(fdim)
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
domain_0_cells = ct.indices[ct.values == vol_ids["omega_0"]]
domain_1_cells = ct.indices[ct.values == vol_ids["omega_1"]]
interface_facets = ft.indices[ft.values == interface]
for facet in interface_facets:
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
        interface_entities.extend(
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

# ext_facet_integration_entities = {boundary_0: []}
boundary_0_entites = []
boundary_facets = ft.indices[ft.values == boundary_0]
for facet in boundary_facets:
    # TODO Remove (bondary facets not shared)
    if facet < facet_imap.size_local:
        cell = f_to_c.links(facet)
        assert len(cell) == 1

        # FIXME Don't use tolist
        local_facet = c_to_f.links(
            cell).tolist().index(facet)

        boundary_0_entites.extend([cell, local_facet])
ds = ufl.Measure("ds", domain=msh,
                 subdomain_data=[(boundary_0, boundary_0_entites)])

# FIXME Do this more efficiently
submesh_0.topology.create_entities(fdim)
submesh_0.topology.create_connectivity(tdim, fdim)
submesh_0.topology.create_connectivity(fdim, tdim)
c_to_f_submesh_0 = submesh_0.topology.connectivity(tdim, fdim)
f_to_c_submesh_0 = submesh_0.topology.connectivity(fdim, tdim)
omega_0_int_entities = []
for facet in range(submesh_0.topology.index_map(fdim).size_local):
    cells = f_to_c_submesh_0.links(facet)
    if len(cells) == 2:
        # FIXME Don't use tolist
        local_facet_plus = c_to_f_submesh_0.links(
            cells[0]).tolist().index(facet)
        local_facet_minus = c_to_f_submesh_0.links(
            cells[1]).tolist().index(facet)

        omega_0_int_entities.extend(
            [entity_map_0[cells[0]], local_facet_plus,
             entity_map_0[cells[1]], local_facet_minus])
dS = ufl.Measure("dS", domain=msh,
                 subdomain_data=[(interface, interface_entities),
                                 (omega_0_int_facets, omega_0_int_entities)])

# TODO Add k dependency
gamma_int = 10  # Penalty param on interface
gamma_dg = 10 * k_0**2  # Penalty parm for DG method
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

x = ufl.SpatialCoordinate(msh)
c = 1.0 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

u_0_n = fem.Function(V_0)
u_1_n = fem.Function(V_1)

w = ufl.as_vector((0.5 - x[1], 0.0))
# w = ufl.as_vector((1e-12, 0.0))
lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), 1, 0)

# TODO Figure out advectve ds term (just integrated over boundary,
# or boundary + interface)
a_00 = inner(u_0 / delta_t, v_0) * dx(vol_ids["omega_0"]) \
    - inner(w * u_0, grad(v_0)) * dx(vol_ids["omega_0"]) \
    + inner(lmbda("+") * dot(w("+"), n("+")) * u_0("+") -
            lmbda("-") * dot(w("-"), n("-")) * u_0("-"),
            jump(v_0)) * dS(omega_0_int_facets) \
    + inner(lmbda * dot(w, n) * u_0, v_0) * ds(boundary_0) + \
    + inner(c * grad(u_0), grad(v_0)) * dx(vol_ids["omega_0"]) \
    - inner(c * avg(grad(u_0)), jump(v_0, n)) * dS(omega_0_int_facets) \
    - inner(c * jump(u_0, n), avg(grad(v_0))) * dS(omega_0_int_facets) \
    + (gamma_dg / avg(h)) * inner(
        c * jump(u_0, n), jump(v_0, n)) * dS(omega_0_int_facets) \
    - inner(c * grad(u_0), v_0 * n) * ds(boundary_0) \
    - inner(c * grad(v_0), u_0 * n) * ds(boundary_0) \
    + (gamma_dg / h) * inner(c * u_0, v_0) * ds(boundary_0) \
    + gamma_int / avg(h) * inner(c * u_0("+"),
                                 v_0("+")) * dS(interface) \
    - inner(c * 1 / 2 * dot(grad(u_0("+")), n("+")),
            v_0("+")) * dS(interface) \
    - inner(c * 1 / 2 * dot(grad(v_0("+")), n("+")),
            u_0("+")) * dS(interface)

a_01 = - gamma_int / avg(h) * inner(c * u_1("-"),
                                    v_0("+")) * dS(interface) \
    + inner(c * 1 / 2 * dot(grad(u_1("-")), n("-")),
            v_0("+")) * dS(interface) \
    + inner(c * 1 / 2 * dot(grad(v_0("+")), n("+")),
            u_1("-")) * dS(interface)

a_10 = - gamma_int / avg(h) * inner(c * u_0("+"),
                                    v_1("-")) * dS(interface) \
    + inner(c * 1 / 2 * dot(grad(u_0("+")), n("+")),
            v_1("-")) * dS(interface) \
    + inner(c * 1 / 2 * dot(grad(v_1("-")), n("-")),
            u_0("+")) * dS(interface)

a_11 = inner(u_1 / delta_t, v_1) * dx(vol_ids["omega_1"]) \
    + inner(c * grad(u_1), grad(v_1)) * dx(vol_ids["omega_1"]) \
    + gamma_int / avg(h) * inner(c * u_1("-"),
                                 v_1("-")) * dS(interface) \
    - inner(c * 1 / 2 * dot(grad(u_1("-")), n("-")),
            v_1("-")) * dS(interface) \
    - inner(c * 1 / 2 * dot(grad(v_1("-")), n("-")),
            u_1("-")) * dS(interface)

a_00 = fem.form(a_00, entity_maps=entity_maps)
a_01 = fem.form(a_01, entity_maps=entity_maps)
a_10 = fem.form(a_10, entity_maps=entity_maps)
a_11 = fem.form(a_11, entity_maps=entity_maps)
a = [[a_00, a_01],
     [a_10, a_11]]


def u_e(x, module=np):
    # return module.exp(- ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / (2 * 0.15**2))
    return module.sin(module.pi * x[0]) * module.sin(module.pi * x[1])


f_0 = dot(w, grad(u_e(ufl.SpatialCoordinate(msh), module=ufl))) \
    - div(c * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl)))
f_1 = - div(c * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl)))

u_D = fem.Function(V_0)
u_D.interpolate(u_e)

L_0 = inner(f_0, v_0) * dx(vol_ids["omega_0"]) \
    - inner((1 - lmbda) * dot(w, n) * u_D, v_0) * ds(boundary_0) \
    + inner(u_0_n / delta_t, v_0) * dx(vol_ids["omega_0"]) \
    - inner(c * u_D * n, grad(v_0)) * ds(boundary_0) \
    + gamma_dg / h * inner(c * u_D, v_0) * ds(boundary_0)
L_1 = inner(f_1, v_1) * dx(vol_ids["omega_1"]) \
    + inner(u_1_n / delta_t, v_1) * dx(vol_ids["omega_1"])

L_0 = fem.form(L_0, entity_maps=entity_maps)
L_1 = fem.form(L_1, entity_maps=entity_maps)
L = [L_0, L_1]


submesh_1_ft = convert_facet_tags(msh, submesh_1, entity_map_1, ft)
bound_facet_sm_1 = submesh_1_ft.indices[
    submesh_1_ft.values == boundary_1]
bound_dofs = fem.locate_dofs_topological(V_1, fdim, bound_facet_sm_1)
u_bc_1 = fem.Function(V_1)
u_bc_1.interpolate(u_e)
bc_1 = fem.dirichletbc(u_bc_1, bound_dofs)
bcs = [bc_1]

A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()

b = fem.petsc.create_vector_block(L)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")

x = A.createVecRight()

u_0_file = io.VTXWriter(msh.comm, "u_0.bp", [u_0_n._cpp_object])
u_1_file = io.VTXWriter(msh.comm, "u_1.bp", [u_1_n._cpp_object])

t = 0.0
u_0_file.write(t)
u_1_file.write(t)
for n in range(num_time_steps):
    t += delta_t

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)

    offset = V_0.dofmap.index_map.size_local * V_0.dofmap.index_map_bs
    u_0_n.x.array[:offset] = x.array_r[:offset]
    u_1_n.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    u_0_n.x.scatter_forward()
    u_1_n.x.scatter_forward()

    u_0_file.write(t)
    u_1_file.write(t)

u_0_file.close()
u_1_file.close()

e_L2_0 = norm_L2(msh.comm, u_0_n - u_e(
    ufl.SpatialCoordinate(submesh_0), module=ufl))
e_L2_1 = norm_L2(msh.comm, u_1_n - u_e(
    ufl.SpatialCoordinate(submesh_1), module=ufl))
e_L2 = np.sqrt(e_L2_0**2 + e_L2_1**2)

if msh.comm.rank == 0:
    print(f"e_L2 = {e_L2}")
