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
V_sm_1 = fem.FunctionSpace(submesh_1, ("Lagrange", 1))
u_sm_1 = fem.Function(V_sm_1)
u_sm_1.interpolate(lambda x: x[1]**2)

# Write the function to file
with io.XDMFFile(comm, "u_sm_1.xdmf", "w") as file:
    file.write_mesh(submesh_1)
    file.write_function(u_sm_1)


W = fem.FunctionSpace(submesh_0, ("Lagrange", 1))

u_sm = ufl.TrialFunction(W)
v_sm = ufl.TestFunction(W)

f_sm = fem.Function(W)
f_sm.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))

submesh_0_facet_imap = submesh_0.topology.index_map(submesh_0_fdim)
submesh_0_num_facets = submesh_0_facet_imap.size_local + \
    submesh_0_facet_imap.num_ghosts
entity_maps_sm = {submesh_1: [entity_map_1.index(entity)
                              if entity in entity_map_1 else -1
                              for entity in range(submesh_0_num_facets)]}

ds_sm = ufl.Measure("ds", domain=submesh_0)
a_sm = fem.form(inner(u_sm, v_sm) * dx + inner(grad(u_sm), grad(v_sm)) * dx)
L_sm = fem.form(inner(f_sm, v_sm) * dx + inner(u_sm_1, v_sm) * ds_sm,
                entity_maps=entity_maps_sm)

A_sm = fem.petsc.assemble_matrix(a_sm)
A_sm.assemble()
b_sm = fem.petsc.assemble_vector(L_sm)
b_sm.ghostUpdate(addv=PETSc.InsertMode.ADD,
                 mode=PETSc.ScatterMode.REVERSE)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A_sm)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

u_sm = fem.Function(W)
ksp.solve(b_sm, u_sm.vector)
u_sm.x.scatter_forward()

with io.XDMFFile(submesh_0.comm, "u_sm.xdmf", "w") as file:
    file.write_mesh(submesh_0)
    file.write_function(u_sm)

num_facets = msh.topology.index_map(msh_fdim).size_local + \
    msh.topology.index_map(msh_fdim).num_ghosts
msh_to_submesh = [entity_map_0.index(entity) if entity in entity_map_0 else -1
                  for entity in range(num_facets)]
entity_maps = {submesh_0: msh_to_submesh}
# print(f"msh_to_submesh = {msh_to_submesh}")

mt = meshtags(msh, msh_fdim, submesh_0_entities,
              np.ones_like(submesh_0_entities))
ds = ufl.Measure("ds", subdomain_data=mt, domain=msh)

V = fem.FunctionSpace(msh, ("Lagrange", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

dirichlet_facets = mesh.locate_entities_boundary(
    msh, msh_fdim, lambda x: np.isclose(x[2], -0.75))
dirichlet_dofs = fem.locate_dofs_topological(V, msh_fdim, dirichlet_facets)
bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dirichlet_dofs, V=V)

f = fem.Function(V)
f.interpolate(lambda x: np.sin(np.pi * x[0])
              * np.sin(np.pi * x[1])
              * np.sin(np.pi * x[2]))

a = fem.form(inner(grad(u), grad(v)) * dx)
L = fem.form(inner(f, v) * dx + inner(u_sm, v) * ds(1),
             entity_maps=entity_maps)
A = fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()

b = fem.petsc.assemble_vector(L)
fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, [bc])

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

u = fem.Function(V)
ksp.solve(b, u.vector)

with io.XDMFFile(msh.comm, "u.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u)
