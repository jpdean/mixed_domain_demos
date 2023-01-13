from dolfinx import mesh, fem
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, avg
import numpy as np

n = 2
msh = mesh.create_rectangle(
    MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
    ghost_mode=mesh.GhostMode.shared_facet)

# Create submeshes of the left and right halves
tdim = msh.topology.dim
left_cells = mesh.locate_entities(msh, tdim, lambda x: x[0] <= 1.0)
left_submesh, left_entity_map = mesh.create_submesh(
    msh, tdim, left_cells)[:2]
right_cells = mesh.locate_entities(msh, tdim, lambda x: x[0] >= 1.0)
right_submesh, right_entity_map = mesh.create_submesh(
    msh, tdim, right_cells)[:2]

cell_integration_entities = {0: left_cells,
                             1: right_cells}
dx = ufl.Measure("dx", domain=msh,
                 subdomain_data=cell_integration_entities)

# Define function spaces on each half
V_0 = fem.FunctionSpace(left_submesh, ("Lagrange", 1))
V_1 = fem.FunctionSpace(right_submesh, ("Lagrange", 1))

# Test and trial functions
u_0 = ufl.TrialFunction(V_0)
u_1 = ufl.TrialFunction(V_1)
v_0 = ufl.TestFunction(V_0)
v_1 = ufl.TestFunction(V_1)

# Get centre facets
fdim = tdim - 1
centre_facets = mesh.locate_entities(
    msh, fdim, lambda x: np.isclose(x[0], 1.0))

# Create entity maps
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
# TODO Replace with more efficient solution
entity_maps = {left_submesh: [left_entity_map.index(entity)
                              if entity in left_entity_map else -1
                              for entity in range(num_cells)],
               right_submesh: [right_entity_map.index(entity)
                               if entity in right_entity_map else -1
                               for entity in range(num_cells)]}

# Create measure for integration. Assign the first (cell, local facet)
# pair to the left cell, corresponding to the "+" restriction. Assign
# the second (cell, local facet) pair to the right cell, corresponding
# to the "-" restriction.
facet_integration_entities = {1: []}
facet_imap = msh.topology.index_map(fdim)
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(fdim, tdim)
c_to_f = msh.topology.connectivity(tdim, fdim)
f_to_c = msh.topology.connectivity(fdim, tdim)
for facet in centre_facets:
    # Check if this facet is owned
    if facet < facet_imap.size_local:
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        cell_plus = cells[0] if cells[0] in left_cells else cells[1]
        cell_minus = cells[0] if cells[0] in right_cells else cells[1]
        assert cell_plus in left_cells
        assert cell_minus in right_cells

        # FIXME Don't use tolist
        local_facet_plus = c_to_f.links(
            cell_plus).tolist().index(facet)
        local_facet_minus = c_to_f.links(
            cell_minus).tolist().index(facet)
        facet_integration_entities[1].extend(
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
        entity_maps[left_submesh][cell_minus] = \
            entity_maps[left_submesh][cell_plus]
        # Same hack for the right submesh
        entity_maps[right_submesh][cell_plus] = \
            entity_maps[right_submesh][cell_minus]
dS = ufl.Measure("dS", domain=msh,
                 subdomain_data=facet_integration_entities)

# TODO Add k dependency
gamma = 10
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

a_00 = inner(grad(u_0), grad(v_0)) * dx(0) \
    + gamma / avg(h) * inner(u_0("+"), v_0("+")) * dS(1) \
    - inner(1 / 2 * dot(grad(u_0("+")), n("+")), v_0("+")) * dS(1) \
    - inner(1 / 2 * dot(grad(v_0("+")), n("+")), u_0("+")) * dS(1)

a_11 = inner(grad(u_1), grad(v_1)) * dx(1) \
    + gamma / avg(h) * inner(u_1("-"), v_1("-")) * dS(1) \
    - inner(1 / 2 * dot(grad(u_1("-")), n("-")), v_1("-")) * dS(1) \
    - inner(1 / 2 * dot(grad(v_1("-")), n("-")), u_1("-")) * dS(1)

a_00 = fem.form(a_00, entity_maps=entity_maps)
a_11 = fem.form(a_11, entity_maps=entity_maps)

A = fem.petsc.assemble_matrix(a_11)
A.assemble()
print(A.norm())

# TODO Pick function that is complicated on one side so most of error
# is there and make that part high-order.
