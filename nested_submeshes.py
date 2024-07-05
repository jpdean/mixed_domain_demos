# This demo shows how nested sub-meshes can be used to solve problems.
# We start from an initial mesh, create a sub-mesh of part of the
# boundary, and then create a sub-mesh of the boundary of the first
# sub-mesh. We then solve a hierarchy of Poisson problems on the meshes.


import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, dx
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
from dolfinx.io import gmshio
from dolfinx.mesh import meshtags, exterior_facet_indices


def create_mesh(comm, h):
    # Create some geometry with gmsh
    gmsh.initialize()
    model = gmsh.model()
    model_name = "Hemisphere"
    if comm.rank == 0:
        # Generate a mesh
        model.add(model_name)
        model.setCurrent(model_name)

        sphere = model.occ.addSphere(0, 0, 0, 1)
        box_0 = model.occ.addBox(-1, -1, 0, 2, 2, 1)
        box_1 = model.occ.addBox(-1, -1, -0.75, 2, 2, -1)
        cylinder = model.occ.addCylinder(0, 0, 0, 0, 0, -1, 0.25)
        cut = model.occ.cut([(3, sphere)], [(3, box_0),
                                            (3, box_1),
                                            (3, cylinder)])
        model.occ.synchronize()

        # Add physical groups
        boundary = model.getBoundary(cut[0], oriented=False)
        boundary_ids = [b[1] for b in boundary]
        model.addPhysicalGroup(2, boundary_ids, tag=1)
        model.setPhysicalName(2, 1, "Sphere surface")

        volume_entities = [model[1] for model in model.getEntities(3)]
        model.addPhysicalGroup(3, volume_entities, tag=2)
        model.setPhysicalName(3, 2, "Sphere volume")

        # # Assign a mesh size to all the points:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

        # Generate the mesh
        model.mesh.generate(3)
        # Use second-order geometry
        model.mesh.setOrder(2)

    msh = gmshio.model_to_mesh(model, comm, 0)[0]
    msh.name = model_name
    return msh


# Create a mesh
comm = MPI.COMM_WORLD
h = 0.25  # Max cell diameter
msh = create_mesh(comm, h)

# Create a sub-mesh of part of the boundary of msh to get a disk
msh_fdim = msh.topology.dim - 1
submesh_0_entities = mesh.locate_entities_boundary(
    msh, msh_fdim, lambda x: np.isclose(x[2], 0.0))
submesh_0, sm_0_to_msh = mesh.create_submesh(
    msh, msh_fdim, submesh_0_entities)[0:2]

# Create a sub-mesh of the boundary of submesh_0 to get concentric circles
submesh_0_tdim = submesh_0.topology.dim
submesh_0_fdim = submesh_0_tdim - 1
submesh_0.topology.create_entities(submesh_0_fdim)
submesh_0.topology.create_connectivity(submesh_0_fdim, submesh_0_tdim)
sm_boundary_facets = exterior_facet_indices(submesh_0.topology)
submesh_1, sm_1_to_sm_0 = mesh.create_submesh(
    submesh_0, submesh_0_fdim, sm_boundary_facets)[0:2]

# Create a functions space on submesh_1 and interpolate a function
k = 2  # Polynomial degree
V_sm_1 = fem.functionspace(submesh_1, ("Lagrange", k))
u_sm_1 = fem.Function(V_sm_1)
u_sm_1.name = "u_sm_1"
u_sm_1.interpolate(lambda x: x[1]**2)

# Write the function to file
with io.VTXWriter(comm, "u_sm_1.bp", u_sm_1, "BP4") as f:
    f.write(0.0)

# Create a function space over submesh_0 and define trial and test
# functions
V_sm_0 = fem.FunctionSpace(submesh_0, ("Lagrange", k))
u_sm_0, v_sm_0 = ufl.TrialFunction(V_sm_0), ufl.TestFunction(V_sm_0)

# Create a function to represent the forcing term
f_sm_0 = fem.Function(V_sm_0)
f_sm_0.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))

# We use submesh_0 as the integration domain mesh, so we must provide a
# map from facets in submesh_0 to cells in submesh_1. This is simply
# the "inverse" of sm_1_to_sm_0
facet_imap_sm_0 = submesh_0.topology.index_map(submesh_0_fdim)
num_facets_sm_0 = facet_imap_sm_0.size_local + \
    facet_imap_sm_0.num_ghosts
sm_0_to_sm_1 = np.full(num_facets_sm_0, -1)
sm_0_to_sm_1[sm_1_to_sm_0] = np.arange(len(sm_1_to_sm_0))
entity_maps_sm_0 = {submesh_1: sm_0_to_sm_1}
ds_sm_0 = ufl.Measure("ds", domain=submesh_0)

# Define forms using the function interpolated on the concentric circle mesh
# as the Neumann boundary condition
a_sm_0 = fem.form(inner(u_sm_0, v_sm_0) * dx +
                  inner(grad(u_sm_0), grad(v_sm_0)) * dx)
L_sm_0 = fem.form(inner(f_sm_0, v_sm_0) * dx + inner(u_sm_1, v_sm_0) * ds_sm_0,
                  entity_maps=entity_maps_sm_0)

# Assemble matrix and vector
A_sm_0 = fem.petsc.assemble_matrix(a_sm_0)
A_sm_0.assemble()
b_sm_0 = fem.petsc.assemble_vector(L_sm_0)
b_sm_0.ghostUpdate(addv=PETSc.InsertMode.ADD,
                   mode=PETSc.ScatterMode.REVERSE)

# Configure solver
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A_sm_0)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Solve
u_sm_0 = fem.Function(V_sm_0)
u_sm_0.name = "u_sm_0"
ksp.solve(b_sm_0, u_sm_0.vector)
u_sm_0.x.scatter_forward()

# Write to file
with io.VTXWriter(comm, "u_sm_0.bp", u_sm_0, "BP4") as f:
    f.write(0.0)

# Create function spaces over the mesh and define trial and test functions
V_msh = fem.FunctionSpace(msh, ("Lagrange", k))
u_msh, v_msh = ufl.TrialFunction(V_msh), ufl.TestFunction(V_msh)

# Create Dirichlet boundary condition
dirichlet_facets = mesh.locate_entities_boundary(
    msh, msh_fdim, lambda x: np.isclose(x[2], -0.75))
dirichlet_dofs = fem.locate_dofs_topological(V_msh, msh_fdim, dirichlet_facets)
bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dirichlet_dofs, V=V_msh)

# Create a function to represent the forcing term
f_msh = fem.Function(V_msh)
f_msh.interpolate(lambda x: np.sin(np.pi * x[0])
                  * np.sin(np.pi * x[1])
                  * np.sin(np.pi * x[2]))

# Create entity maps
num_facets_msh = msh.topology.index_map(msh_fdim).size_local + \
    msh.topology.index_map(msh_fdim).num_ghosts
msh_to_sm_0 = np.full(num_facets_msh, -1)
msh_to_sm_0[sm_0_to_msh] = np.arange(len(sm_0_to_msh))
entity_maps_msh = {submesh_0: msh_to_sm_0}

# Create meshtags to mark the Neumann boundary
mt = meshtags(msh, msh_fdim, submesh_0_entities,
              np.ones_like(submesh_0_entities))
ds_msh = ufl.Measure("ds", subdomain_data=mt, domain=msh)

a_msh = fem.form(inner(grad(u_msh), grad(v_msh)) * dx)
L_msh = fem.form(inner(f_msh, v_msh) * dx + inner(u_sm_0, v_msh) * ds_msh(1),
                 entity_maps=entity_maps_msh)

# Assemble matrix and vector
A_msh = fem.petsc.assemble_matrix(a_msh, bcs=[bc])
A_msh.assemble()
b_msh = fem.petsc.assemble_vector(L_msh)
fem.petsc.apply_lifting(b_msh, [a_msh], bcs=[[bc]])
b_msh.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b_msh, [bc])

# Solve
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A_msh)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

u_msh = fem.Function(V_msh)
u_msh.name = "u_msh"
ksp.solve(b_msh, u_msh.vector)
u_msh.x.scatter_forward()

# Write to file
with io.VTXWriter(comm, "u_msh.bp", u_msh, "BP4") as f:
    f.write(0.0)
