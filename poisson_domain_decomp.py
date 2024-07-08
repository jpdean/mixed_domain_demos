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
from utils import norm_L2, convert_facet_tags, compute_interface_integration_entities
import gmsh
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block


def u_e(x, module=np):
    "A function to represent the exact solution"
    return module.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / (2 * 0.05**2)) + x[0]


def create_mesh(h, c=0.5, r=0.25):
    """
    Create a mesh of a square domain. The mesh conforms with the boundary of a
    circle inscribed inside the domain with centre c and radius r.

    Parameters:
        h: maximum cell diameter
        c: centre of the inscribed circle
        r: radius of the circle

    Returns:
        A mesh, cell tags, and facet tags
    """
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("square_with_circle")

        factory = gmsh.model.geo

        # Corners of the square
        square_points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 1.0, 0.0, h),
            factory.addPoint(0.0, 1.0, 0.0, h),
        ]

        # The centre of the circle and four points lying on its boundary.
        circle_points = [
            factory.addPoint(c, c, 0.0, h),
            factory.addPoint(c + r, c, 0.0, h),
            factory.addPoint(c, c + r, 0.0, h),
            factory.addPoint(c - r, c, 0.0, h),
            factory.addPoint(c, c - r, 0.0, h),
        ]

        # The boundary of the square
        square_lines = [
            factory.addLine(square_points[0], square_points[1]),
            factory.addLine(square_points[1], square_points[2]),
            factory.addLine(square_points[2], square_points[3]),
            factory.addLine(square_points[3], square_points[0]),
        ]

        # The boundary of the circle
        circle_lines = [
            factory.addCircleArc(circle_points[1], circle_points[0], circle_points[2]),
            factory.addCircleArc(circle_points[2], circle_points[0], circle_points[3]),
            factory.addCircleArc(circle_points[3], circle_points[0], circle_points[4]),
            factory.addCircleArc(circle_points[4], circle_points[0], circle_points[1]),
        ]

        # Create curves
        square_curve = factory.addCurveLoop(square_lines)
        circle_curve = factory.addCurveLoop(circle_lines)

        # Create surfaces
        square_surface = factory.addPlaneSurface([square_curve, circle_curve])
        circle_surface = factory.addPlaneSurface([circle_curve])

        factory.synchronize()

        # Tag physical groups
        gmsh.model.addPhysicalGroup(2, [square_surface], vol_ids["omega_0"])
        gmsh.model.addPhysicalGroup(2, [circle_surface], vol_ids["omega_1"])
        gmsh.model.addPhysicalGroup(1, square_lines, surf_ids["boundary"])
        gmsh.model.addPhysicalGroup(1, circle_lines, surf_ids["interface"])

        gmsh.model.mesh.generate(2)

        # gmsh.fltk.run()

    # Create dolfinx mesh
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, ct, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=2, partitioner=partitioner
    )
    gmsh.finalize()
    return msh, ct, ft


# Set some parameters
comm = MPI.COMM_WORLD
h = 0.05  # Maximum cell diameter
k_0 = 1  # Polynomial degree in omega_0
k_1 = 3  # Polynomial degree in omega_1

# Tags for volumes and surfaces
vol_ids = {"omega_0": 1, "omega_1": 2}
surf_ids = {"boundary": 3, "interface": 4}

# Create mesh and sub-meshes
msh, ct, ft = create_mesh(h)
tdim = msh.topology.dim
submesh_0, sm_0_to_msh = mesh.create_submesh(msh, tdim, ct.find(vol_ids["omega_0"]))[:2]
submesh_1, sm_1_to_msh = mesh.create_submesh(msh, tdim, ct.find(vol_ids["omega_1"]))[:2]

# Define function spaces on each submesh
V_0 = fem.functionspace(submesh_0, ("Lagrange", k_0))
V_1 = fem.functionspace(submesh_1, ("Lagrange", k_1))

# Test and trial functions
u_0, u_1 = ufl.TrialFunction(V_0), ufl.TrialFunction(V_1)
v_0, v_1 = ufl.TestFunction(V_0), ufl.TestFunction(V_1)

# We use msh as the integration domain, so we require maps from cells
# in msh to cells in submesh_0 and submesh_1. These can be created
# as follows:
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
msh_to_sm_0 = np.full(num_cells, -1)
msh_to_sm_0[sm_0_to_msh] = np.arange(len(sm_0_to_msh))
msh_to_sm_1 = np.full(num_cells, -1)
msh_to_sm_1[sm_1_to_msh] = np.arange(len(sm_1_to_msh))
entity_maps = {submesh_0: msh_to_sm_0, submesh_1: msh_to_sm_1}

# Compute integration entities for the interface integral
fdim = tdim - 1
interface_facets = ft.find(surf_ids["interface"])
domain_0_cells = ct.find(vol_ids["omega_0"])
domain_1_cells = ct.find(vol_ids["omega_1"])
interface_entities, msh_to_sm_0, msh_to_sm_1 = compute_interface_integration_entities(
    msh, interface_facets, domain_0_cells, domain_1_cells, msh_to_sm_0, msh_to_sm_1
)

# Create integration measures
dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
dS = ufl.Measure(
    "dS", domain=msh, subdomain_data=[(surf_ids["interface"], interface_entities)]
)

# TODO Add k dependency
gamma = 10  # Penalty parameter
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

x = ufl.SpatialCoordinate(msh)
c = 1.0 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

# Define left-hand side forms
domain_0, domain_1 = "+", "-"

a_00 = (
    inner(c * grad(u_0), grad(v_0)) * dx(vol_ids["omega_0"])
    + gamma
    / avg(h)
    * inner(c * u_0(domain_0), v_0(domain_0))
    * dS(surf_ids["interface"])
    - inner(c * 1 / 2 * dot(grad(u_0(domain_0)), n(domain_0)), v_0(domain_0))
    * dS(surf_ids["interface"])
    - inner(c * 1 / 2 * dot(grad(v_0(domain_0)), n(domain_0)), u_0(domain_0))
    * dS(surf_ids["interface"])
)

a_01 = (
    -gamma
    / avg(h)
    * inner(c * u_1(domain_1), v_0(domain_0))
    * dS(surf_ids["interface"])
    + inner(c * 1 / 2 * dot(grad(u_1(domain_1)), n(domain_1)), v_0(domain_0))
    * dS(surf_ids["interface"])
    + inner(c * 1 / 2 * dot(grad(v_0(domain_0)), n(domain_0)), u_1(domain_1))
    * dS(surf_ids["interface"])
)

a_10 = (
    -gamma
    / avg(h)
    * inner(c * u_0(domain_0), v_1(domain_1))
    * dS(surf_ids["interface"])
    + inner(c * 1 / 2 * dot(grad(u_0(domain_0)), n(domain_0)), v_1(domain_1))
    * dS(surf_ids["interface"])
    + inner(c * 1 / 2 * dot(grad(v_1(domain_1)), n(domain_1)), u_0(domain_0))
    * dS(surf_ids["interface"])
)

a_11 = (
    inner(c * grad(u_1), grad(v_1)) * dx(vol_ids["omega_1"])
    + gamma
    / avg(h)
    * inner(c * u_1(domain_1), v_1(domain_1))
    * dS(surf_ids["interface"])
    - inner(c * 1 / 2 * dot(grad(u_1(domain_1)), n(domain_1)), v_1(domain_1))
    * dS(surf_ids["interface"])
    - inner(c * 1 / 2 * dot(grad(v_1(domain_1)), n(domain_1)), u_1(domain_1))
    * dS(surf_ids["interface"])
)

# Compile LHS forms and set block structure
a_00 = fem.form(a_00, entity_maps=entity_maps)
a_01 = fem.form(a_01, entity_maps=entity_maps)
a_10 = fem.form(a_10, entity_maps=entity_maps)
a_11 = fem.form(a_11, entity_maps=entity_maps)
a = [[a_00, a_01], [a_10, a_11]]

# Define right-hand side forms
f = -div(c * grad(u_e(ufl.SpatialCoordinate(msh), module=ufl)))
L_0 = inner(f, v_0) * dx(vol_ids["omega_0"])
L_1 = inner(f, v_1) * dx(vol_ids["omega_1"])

# Compile RHS forms and set block structure
L_0 = fem.form(L_0, entity_maps=entity_maps)
L_1 = fem.form(L_1, entity_maps=entity_maps)
L = [L_0, L_1]

# Apply boundary conditions. We require the DOFs of V_0 on the domain
# boundary. These can be identified via that facets of submesh_0 that
# lie on the domain boundary.
ft_sm_0 = convert_facet_tags(msh, submesh_0, sm_0_to_msh, ft)
bound_facets_sm_0 = ft_sm_0.find(surf_ids["boundary"])
submesh_0.topology.create_connectivity(fdim, tdim)
bound_dofs = fem.locate_dofs_topological(V_0, fdim, bound_facets_sm_0)
u_bc_0 = fem.Function(V_0)
u_bc_0.interpolate(u_e)
bc_0 = fem.dirichletbc(u_bc_0, bound_dofs)
bcs = [bc_0]

# Assemble linear system of equations
A = assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = assemble_vector_block(L, a, bcs=bcs)

# Set-up solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

# Recover solution
u_0, u_1 = fem.Function(V_0), fem.Function(V_1)
offset = V_0.dofmap.index_map.size_local * V_0.dofmap.index_map_bs
u_0.x.array[:offset] = x.array_r[:offset]
u_1.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
u_0.x.scatter_forward()
u_1.x.scatter_forward()

# Write solution to file
with io.VTXWriter(msh.comm, "u_0.bp", u_0, "BP4") as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "u_1.bp", u_1, "BP4") as f:
    f.write(0.0)

# Compute error in solution
e_L2_0 = norm_L2(msh.comm, u_0 - u_e(ufl.SpatialCoordinate(submesh_0), module=ufl))
e_L2_1 = norm_L2(msh.comm, u_1 - u_e(ufl.SpatialCoordinate(submesh_1), module=ufl))
e_L2 = np.sqrt(e_L2_0**2 + e_L2_1**2)

if msh.comm.rank == 0:
    print(e_L2)
