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
from utils import (
    norm_L2,
    convert_facet_tags,
    compute_interface_integration_entities,
    compute_interior_facet_integration_entities,
)
from dolfinx.cpp.fem import compute_integration_domains
import gmsh
from dolfinx.fem.petsc import (
    assemble_matrix_block,
    create_vector_block,
    assemble_vector_block,
)


def u_e(x, module=np):
    "Function to represent the exact solution"
    # return module.exp(- ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / (2 * 0.15**2))
    return module.sin(module.pi * x[0]) * module.sin(module.pi * x[1])


def create_mesh(comm, h):
    "Create a mesh of the unit square divided into two regions"
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("model")
        factory = gmsh.model.geo

        # Create points at each corner of the square, and add
        # additional points at the midpoint of the vertical sides
        # of the square
        points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 0.5, 0.0, h),
            factory.addPoint(0.0, 0.5, 0.0, h),
            factory.addPoint(0.0, 1.0, 0.0, h),
            factory.addPoint(1.0, 1.0, 0.0, h),
        ]

        # Lines bounding omega_0
        omega_0_lines = [
            factory.addLine(points[0], points[1]),
            factory.addLine(points[1], points[2]),
            factory.addLine(points[2], points[3]),
            factory.addLine(points[3], points[0]),
        ]

        # Lines bounding omega_1
        omega_1_lines = [
            omega_0_lines[2],
            factory.addLine(points[3], points[4]),
            factory.addLine(points[4], points[5]),
            factory.addLine(points[5], points[2]),
        ]

        # Create curves and surfaces
        omega_0_curve = factory.addCurveLoop(omega_0_lines)
        omega_1_curve = factory.addCurveLoop(omega_1_lines)
        omega_0_surface = factory.addPlaneSurface([omega_0_curve])
        omega_1_surface = factory.addPlaneSurface([omega_1_curve])

        factory.synchronize()

        # Add physical groups
        gmsh.model.addPhysicalGroup(2, [omega_0_surface], vol_ids["omega_0"])
        gmsh.model.addPhysicalGroup(2, [omega_1_surface], vol_ids["omega_1"])
        gmsh.model.addPhysicalGroup(
            1,
            [omega_0_lines[0], omega_0_lines[1], omega_0_lines[3]],
            bound_ids["boundary_0"],
        )
        gmsh.model.addPhysicalGroup(
            1,
            [omega_1_lines[1], omega_1_lines[2], omega_1_lines[3]],
            bound_ids["boundary_1"],
        )
        gmsh.model.addPhysicalGroup(1, [omega_0_lines[2]], bound_ids["interface"])

        # Generate mesh
        gmsh.model.mesh.generate(2)
        # gmsh.fltk.run()

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, ct, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=2, partitioner=partitioner
    )
    gmsh.finalize()
    return msh, ct, ft


# Set some parameters
num_time_steps = 10
k_0 = 3  # Polynomial degree in omega_0
k_1 = 3  # Polynomial degree in omgea_1
delta_t = 1  # TODO Make constant
h = 0.05  # Maximum cell diameter

# Volume and boundary ids
vol_ids = {"omega_0": 1, "omega_1": 2}
bound_ids = {
    "boundary_0": 3,
    "boundary_1": 4,
    "interface": 5,  # Interface of omega_0
    "omega_0_int_facets": 6,
}  # Interior facets of omega_0

# Create the mesh
comm = MPI.COMM_WORLD
msh, ct, ft = create_mesh(comm, h)

# Create sub-meshes of omega_0 and omega_1 so that we can create
# different function spaces over each part of the domain
tdim = msh.topology.dim
submesh_0, sm_0_to_msh = mesh.create_submesh(msh, tdim, ct.find(vol_ids["omega_0"]))[:2]
submesh_1, sm_1_to_msh = mesh.create_submesh(msh, tdim, ct.find(vol_ids["omega_1"]))[:2]

# Define function spaces on each submesh
V_0 = fem.functionspace(submesh_0, ("Discontinuous Lagrange", k_0))
V_1 = fem.functionspace(submesh_1, ("Lagrange", k_1))

# Test and trial functions
u_0, v_0 = ufl.TrialFunction(V_0), ufl.TestFunction(V_0)
u_1, v_1 = ufl.TrialFunction(V_1), ufl.TestFunction(V_1)

# We use msh as the integration domain, so we require maps from
# cells in msh to cells in submesh_0 and submesh_1
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
msh_to_sm_0 = np.full(num_cells, -1)
msh_to_sm_0[sm_0_to_msh] = np.arange(len(sm_0_to_msh))
msh_to_sm_1 = np.full(num_cells, -1)
msh_to_sm_1[sm_1_to_msh] = np.arange(len(sm_1_to_msh))

# Create integration entities for the interface integral
interface_facets = ft.find(bound_ids["interface"])
domain_0_cells = ct.find(vol_ids["omega_0"])
domain_1_cells = ct.find(vol_ids["omega_1"])
interface_entities, msh_to_sm_0, msh_to_sm_1 = compute_interface_integration_entities(
    msh, interface_facets, domain_0_cells, domain_1_cells, msh_to_sm_0, msh_to_sm_1
)

# Compute integration entities for boundary terms
boundary_entites = [
    (
        bound_ids["boundary_0"],
        compute_integration_domains(
            fem.IntegralType.exterior_facet,
            msh.topology,
            ft.find(bound_ids["boundary_0"]),
            ft.dim,
        ),
    )
]

# Compute integration entities for the interior facet integrals
# over omega_0. These are needed for the DG scheme
omega_0_int_entities = compute_interior_facet_integration_entities(
    submesh_0, sm_0_to_msh
)

# Create measures
dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
ds = ufl.Measure("ds", domain=msh, subdomain_data=boundary_entites)
dS = ufl.Measure(
    "dS",
    domain=msh,
    subdomain_data=[
        (bound_ids["interface"], interface_entities),
        (bound_ids["omega_0_int_facets"], omega_0_int_entities),
    ],
)

# Define forms
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

# Forms for the left-had side
a_00 = (
    inner(u_0 / delta_t, v_0) * dx(vol_ids["omega_0"])
    - inner(w * u_0, grad(v_0)) * dx(vol_ids["omega_0"])
    + inner(
        lmbda("+") * dot(w("+"), n("+")) * u_0("+")
        - lmbda("-") * dot(w("-"), n("-")) * u_0("-"),
        jump(v_0),
    )
    * dS(bound_ids["omega_0_int_facets"])
    + inner(lmbda * dot(w, n) * u_0, v_0) * ds(bound_ids["boundary_0"])
    + +inner(c * grad(u_0), grad(v_0)) * dx(vol_ids["omega_0"])
    - inner(c * avg(grad(u_0)), jump(v_0, n)) * dS(bound_ids["omega_0_int_facets"])
    - inner(c * jump(u_0, n), avg(grad(v_0))) * dS(bound_ids["omega_0_int_facets"])
    + (gamma_dg / avg(h))
    * inner(c * jump(u_0, n), jump(v_0, n))
    * dS(bound_ids["omega_0_int_facets"])
    - inner(c * grad(u_0), v_0 * n) * ds(bound_ids["boundary_0"])
    - inner(c * grad(v_0), u_0 * n) * ds(bound_ids["boundary_0"])
    + (gamma_dg / h) * inner(c * u_0, v_0) * ds(bound_ids["boundary_0"])
    + gamma_int / avg(h) * inner(c * u_0("+"), v_0("+")) * dS(bound_ids["interface"])
    - inner(c * 1 / 2 * dot(grad(u_0("+")), n("+")), v_0("+"))
    * dS(bound_ids["interface"])
    - inner(c * 1 / 2 * dot(grad(v_0("+")), n("+")), u_0("+"))
    * dS(bound_ids["interface"])
)

a_01 = (
    -gamma_int / avg(h) * inner(c * u_1("-"), v_0("+")) * dS(bound_ids["interface"])
    + inner(c * 1 / 2 * dot(grad(u_1("-")), n("-")), v_0("+"))
    * dS(bound_ids["interface"])
    + inner(c * 1 / 2 * dot(grad(v_0("+")), n("+")), u_1("-"))
    * dS(bound_ids["interface"])
)

a_10 = (
    -gamma_int / avg(h) * inner(c * u_0("+"), v_1("-")) * dS(bound_ids["interface"])
    + inner(c * 1 / 2 * dot(grad(u_0("+")), n("+")), v_1("-"))
    * dS(bound_ids["interface"])
    + inner(c * 1 / 2 * dot(grad(v_1("-")), n("-")), u_0("+"))
    * dS(bound_ids["interface"])
)

a_11 = (
    inner(u_1 / delta_t, v_1) * dx(vol_ids["omega_1"])
    + inner(c * grad(u_1), grad(v_1)) * dx(vol_ids["omega_1"])
    + gamma_int / avg(h) * inner(c * u_1("-"), v_1("-")) * dS(bound_ids["interface"])
    - inner(c * 1 / 2 * dot(grad(u_1("-")), n("-")), v_1("-"))
    * dS(bound_ids["interface"])
    - inner(c * 1 / 2 * dot(grad(v_1("-")), n("-")), u_1("-"))
    * dS(bound_ids["interface"])
)

# Compile LHS forms
entity_maps = {submesh_0: msh_to_sm_0, submesh_1: msh_to_sm_1}
a_00 = fem.form(a_00, entity_maps=entity_maps)
a_01 = fem.form(a_01, entity_maps=entity_maps)
a_10 = fem.form(a_10, entity_maps=entity_maps)
a_11 = fem.form(a_11, entity_maps=entity_maps)
a = [[a_00, a_01], [a_10, a_11]]

# Forms for the righ-hand side
f_0 = dot(w, grad(u_e(ufl.SpatialCoordinate(msh), module=ufl))) - div(
    c * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl))
)
f_1 = -div(c * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl)))

u_D = fem.Function(V_0)
u_D.interpolate(u_e)

L_0 = (
    inner(f_0, v_0) * dx(vol_ids["omega_0"])
    - inner((1 - lmbda) * dot(w, n) * u_D, v_0) * ds(bound_ids["boundary_0"])
    + inner(u_0_n / delta_t, v_0) * dx(vol_ids["omega_0"])
    - inner(c * u_D * n, grad(v_0)) * ds(bound_ids["boundary_0"])
    + gamma_dg / h * inner(c * u_D, v_0) * ds(bound_ids["boundary_0"])
)
L_1 = inner(f_1, v_1) * dx(vol_ids["omega_1"]) + inner(u_1_n / delta_t, v_1) * dx(
    vol_ids["omega_1"]
)

# Compile RHS forms
L_0 = fem.form(L_0, entity_maps=entity_maps)
L_1 = fem.form(L_1, entity_maps=entity_maps)
L = [L_0, L_1]

# Apply boundary condition. Since the boundary condition is applied on
# V_1, we must convert the facet tags to submesh_1 in order to locate
# the boundary degrees of freedom.
# NOTE: We don't do this for V_0 since the Dirichlet boundary condition
# is enforced weakly by the DG scheme.
ft_sm_1 = convert_facet_tags(msh, submesh_1, sm_1_to_msh, ft)
bound_facets_sm_1 = ft_sm_1.find(bound_ids["boundary_1"])
submesh_1.topology.create_connectivity(tdim - 1, tdim)
bound_dofs = fem.locate_dofs_topological(V_1, tdim - 1, bound_facets_sm_1)
u_bc_1 = fem.Function(V_1)
u_bc_1.interpolate(u_e)
bc_1 = fem.dirichletbc(u_bc_1, bound_dofs)
bcs = [bc_1]

# Assemble the system of equations
A = assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = create_vector_block(L)

# Set up solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")

# Setup files for visualisation
u_0_file = io.VTXWriter(msh.comm, "u_0.bp", [u_0_n._cpp_object], "BP4")
u_1_file = io.VTXWriter(msh.comm, "u_1.bp", [u_1_n._cpp_object], "BP4")

# Time stepping loop
t = 0.0
u_0_file.write(t)
u_1_file.write(t)
x = A.createVecRight()
for n in range(num_time_steps):
    t += delta_t

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector_block(b, L, a, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)

    # Recover solution
    offset = V_0.dofmap.index_map.size_local * V_0.dofmap.index_map_bs
    u_0_n.x.array[:offset] = x.array_r[:offset]
    u_1_n.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
    u_0_n.x.scatter_forward()
    u_1_n.x.scatter_forward()

    # Write to file
    u_0_file.write(t)
    u_1_file.write(t)

u_0_file.close()
u_1_file.close()

# Compute errors
e_L2_0 = norm_L2(msh.comm, u_0_n - u_e(ufl.SpatialCoordinate(submesh_0), module=ufl))
e_L2_1 = norm_L2(msh.comm, u_1_n - u_e(ufl.SpatialCoordinate(submesh_1), module=ufl))
e_L2 = np.sqrt(e_L2_0**2 + e_L2_1**2)

if msh.comm.rank == 0:
    print(f"e_L2 = {e_L2}")
