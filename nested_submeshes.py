import numpy as np
import ufl
from dolfinx import fem, io, mesh
from ufl import grad, inner, dx
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
from dolfinx.io import gmshio
from dolfinx.mesh import meshtags, exterior_facet_indices


# Create some geometry with gmsh
gmsh.initialize()
comm = MPI.COMM_WORLD
model_rank = 0
model = gmsh.model()
model_name = "Hemisphere"
if comm.rank == model_rank:
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

    # Add physical tag 1 for exterior surfaces
    boundary = model.getBoundary(cut[0], oriented=False)
    boundary_ids = [b[1] for b in boundary]
    model.addPhysicalGroup(2, boundary_ids, tag=1)
    model.setPhysicalName(2, 1, "Sphere surface")

    # Add physical tag 2 for the volume
    volume_entities = [model[1] for model in model.getEntities(3)]
    model.addPhysicalGroup(3, volume_entities, tag=2)
    model.setPhysicalName(3, 2, "Sphere volume")

    # Generate the mesh
    model.mesh.generate(3)
    # Use second-order geometry
    model.mesh.setOrder(2)

msh = gmshio.model_to_mesh(model, comm, model_rank)[0]
msh.name = model_name

# Create a submesh of part of the boundary of the original mesh to
# get a disk
msh_fdim = msh.topology.dim - 1
submesh_0_entities = mesh.locate_entities_boundary(
    msh, msh_fdim, lambda x: np.isclose(x[2], 0.0))
submesh_0, entity_map_0 = mesh.create_submesh(
    msh, msh_fdim, submesh_0_entities)[0:2]

# Create a submesh of the boundary of submesh_0 to get some concentric
# circles
submesh_0_tdim = submesh_0.topology.dim
submesh_0_fdim = submesh_0_tdim - 1
submesh_0.topology.create_entities(submesh_0_fdim)
submesh_0.topology.create_connectivity(submesh_0_fdim, submesh_0_tdim)
sm_boundary_facets = exterior_facet_indices(submesh_0.topology)
submesh_1, entity_map_1 = mesh.create_submesh(
    submesh_0, submesh_0_fdim, sm_boundary_facets)[0:2]

# Create a functions space on submesh_1 and interpolate a function
k = 2
V_sm_1 = fem.FunctionSpace(submesh_1, ("Lagrange", k))
u_sm_1 = fem.Function(V_sm_1)
u_sm_1.name = "u_sm_1"
u_sm_1.interpolate(lambda x: x[1]**2)

# Write the function to file
with io.VTXWriter(comm, "u_sm_1.bp", u_sm_1) as f:
    f.write(0.0)

# Create a function space over submesh_0, and define trial and test
# functions
V_sm_0 = fem.FunctionSpace(submesh_0, ("Lagrange", k))
u_sm_0 = ufl.TrialFunction(V_sm_0)
v_sm_0 = ufl.TestFunction(V_sm_0)

# Create a function to represent the forcing term
f_sm_0 = fem.Function(V_sm_0)
f_sm_0.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))

# Create the entity maps and an integration measure
submesh_0_facet_imap = submesh_0.topology.index_map(submesh_0_fdim)
submesh_0_num_facets = submesh_0_facet_imap.size_local + \
    submesh_0_facet_imap.num_ghosts
entity_maps_sm_0 = {submesh_1: [entity_map_1.index(entity)
                                if entity in entity_map_1 else -1
                                for entity in range(submesh_0_num_facets)]}
ds_sm_0 = ufl.Measure("ds", domain=submesh_0)

# Define forms, using the function interpolated on the concentric circle mesh
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
with io.VTXWriter(comm, "u_sm_0.bp", u_sm_0) as f:
    f.write(0.0)

# Create function spaces over the mesh, and define trial and test functions
V_msh = fem.FunctionSpace(msh, ("Lagrange", k))
u_msh = ufl.TrialFunction(V_msh)
v_msh = ufl.TestFunction(V_msh)

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
msh_num_facets = msh.topology.index_map(msh_fdim).size_local + \
    msh.topology.index_map(msh_fdim).num_ghosts
entity_maps_msh = {submesh_0: [entity_map_0.index(entity) if entity in entity_map_0 else -1
                               for entity in range(msh_num_facets)]}

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
with io.VTXWriter(comm, "u_msh.bp", u_msh) as f:
    f.write(0.0)