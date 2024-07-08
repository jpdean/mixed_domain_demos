import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh
import sys
from dolfinx.cpp.mesh import cell_num_entities


def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def norm_L2(comm, v, measure=ufl.dx):
    return np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(v, v) * measure)), op=MPI.SUM
        )
    )


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(
            fem.form(fem.Constant(msh, PETSc.ScalarType(1.0)) * ufl.dx)
        ),
        op=MPI.SUM,
    )
    return (
        1
        / vol
        * msh.comm.allreduce(fem.assemble_scalar(fem.form(v * ufl.dx)), op=MPI.SUM)
    )


def normal_jump_error(msh, v):
    n = ufl.FacetNormal(msh)
    return norm_L2(msh.comm, ufl.jump(v, n), measure=ufl.dS)


# FIXME This should be a C++ helper function
# TODO Simplify, make scalable, and document
def convert_facet_tags(msh, submesh, cell_map, facet_tag):
    msh_facets = facet_tag.indices

    # Connectivities
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, tdim - 1)
    msh.topology.create_connectivity(tdim - 1, tdim)
    msh_c_to_f = msh.topology.connectivity(tdim, tdim - 1)
    msh_f_to_c = msh.topology.connectivity(tdim - 1, tdim)
    submesh.topology.create_connectivity(tdim, tdim - 1)
    submesh_c_to_f = submesh.topology.connectivity(tdim, tdim - 1)

    # NOTE: Tagged facets mat not have a cell in the submesh, or may
    # have more than one cell in the submesh
    submesh_facets = []
    submesh_values = []
    for i, facet in enumerate(msh_facets):
        cells = msh_f_to_c.links(facet)
        for cell in cells:
            if cell in cell_map:
                local_facet = msh_c_to_f.links(cell).tolist().index(facet)
                # FIXME Don't hardcode cell type
                assert local_facet >= 0  # and local_facet <= 2
                submesh_cell = np.where(cell_map == cell)[0][0]
                submesh_facet = submesh_c_to_f.links(submesh_cell)[local_facet]
                submesh_facets.append(submesh_facet)
                submesh_values.append(facet_tag.values[i])
    submesh_facets = np.array(submesh_facets)
    submesh_values = np.array(submesh_values, dtype=np.intc)
    # Sort and make unique
    submesh_facets, ind = np.unique(submesh_facets, return_index=True)
    submesh_values = submesh_values[ind]
    submesh_meshtags = mesh.meshtags(
        submesh, submesh.topology.dim - 1, submesh_facets, submesh_values
    )
    return submesh_meshtags


def create_random_mesh(corners, n, ghost_mode):
    """Create a rectangular mesh made of randomly ordered simplices"""
    if MPI.COMM_WORLD.rank == 0:
        h_x = (corners[1][0] - corners[0][0]) / n[0]
        h_y = (corners[1][1] - corners[0][1]) / n[1]

        points = [(i * h_x, j * h_y) for i in range(n[0] + 1) for j in range(n[1] + 1)]

        import random

        random.seed(6)

        cells = []
        for i in range(n[0]):
            for j in range(n[1]):
                v = (n[1] + 1) * i + j
                cell_0 = [v, v + 1, v + n[1] + 2]
                random.shuffle(cell_0)
                cells.append(cell_0)

                cell_1 = [v, v + n[1] + 1, v + n[1] + 2]
                random.shuffle(cell_1)
                cells.append(cell_1)
        cells = np.array(cells)
        points = np.array(points)
    else:
        cells, points = np.empty([0, 3]), np.empty([0, 2])

    import basix.ufl_wrapper

    domain = ufl.Mesh(
        basix.ufl_wrapper.create_vector_element("Lagrange", "triangle", 1)
    )
    partitioner = mesh.create_cell_partitioner(ghost_mode)
    return mesh.create_mesh(
        MPI.COMM_WORLD, cells, points, domain, partitioner=partitioner
    )


def create_trap_mesh(
    comm, n, corners, offset_scale=0.25, ghost_mode=mesh.GhostMode.none
):
    """Creates a trapezium mesh by creating a square mesh and offsetting
    the points by a fraction of the cell diameter. The offset can be
    controlled with offset_scale.
    Parameters:
        n: Number of elements in each direction
        corners: coordinates of the bottom left and upper right corners
        offset_scale: Fraction of cell diameter to offset the points
        ghost_mode: The ghost mode
    Returns:
        mesh: A dolfinx mesh object
    """
    if n[1] % 2 != 0:
        raise Exception("n[1] must be even")

    if comm.rank == 0:
        # Width of each element
        h = [(corners[1][i] - corners[0][i]) / n[i] for i in range(2)]

        x = []
        for j in range(n[1] + 1):
            for i in range(n[0] + 1):
                offset = 0
                if j % 2 != 0:
                    if i % 2 == 0:
                        offset = offset_scale * h[1]
                    else:
                        offset = -offset_scale * h[1]
                x.append([corners[0][0] + i * h[0], corners[0][1] + j * h[1] + offset])
        x = np.array(x)

        cells = []
        for j in range(n[1]):
            for i in range(n[0]):
                node_0 = i + (n[0] + 1) * j
                node_1 = i + (n[0] + 1) * j + 1
                node_2 = i + (n[0] + 1) * (j + 1)
                node_3 = i + (n[0] + 1) * (j + 1) + 1

                cells.append([node_0, node_1, node_2, node_3])
        cells = np.array(cells)
    else:
        cells, x = np.empty([0, 3]), np.empty([0, 2])

    ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    partitioner = mesh.create_cell_partitioner(ghost_mode)
    msh = mesh.create_mesh(comm, cells, x, ufl_mesh, partitioner=partitioner)

    return msh


class TimeDependentExpression:
    """Simple class to represent time dependent functions"""

    def __init__(self, expression):
        self.t = 0
        self.expression = expression

    def __call__(self, x):
        return self.expression(x, self.t)


def compute_interface_integration_entities(
    msh,
    interface_facets,
    domain_0_cells,
    domain_1_cells,
    domain_to_domain_0,
    domain_to_domain_1,
):
    """
    This function computes the integration entities (as a list of pairs of
    (cell, local facet index) pairs) required to assemble mixed domain forms
    over the interface. It assumes there is a domain with two sub-domains,
    domain_0 and domain_1, that have a common interface. Cells in domain_0
    correspond to the "-" restriction and cells in domain_1 correspond to
    the "-" restriction.

    Parameters:
        interface_facets: A list of facets on the interface
        domain_0_cells: A list of cells in domain_0
        domain_1_cells: A list of cells in domain_1
        c_to_f: The cell to facet connectivity for the domain mesh
        f_to_c: the facet to cell connectivity for the domain mesh
        facet_imap: The facet index_map for the domain mesh
        domain_to_domain_0: A map from cells in domain to cells in domain_0
        domain_to_domain_1: A map from cells in domain to cells in domain_1

    Returns:
        interface_entities: The integration entities
        domain_to_domain_0: A modified map (see HACK below)
        domain_to_domain_1: A modified map (see HACK below)
    """
    # Create measure for integration. Assign the first (cell, local facet)
    # pair to the cell in domain_0, corresponding to the "+" restriction.
    # Assign the second pair to the domain_1 cell, corresponding to the "-"
    # restriction.
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    facet_imap = msh.topology.index_map(fdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    # FIXME This can be done more efficiently
    interface_entities = []
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
            local_facet_plus = c_to_f.links(cell_plus).tolist().index(facet)
            local_facet_minus = c_to_f.links(cell_minus).tolist().index(facet)
            interface_entities.extend(
                [cell_plus, local_facet_plus, cell_minus, local_facet_minus]
            )

            # FIXME HACK cell_minus does not exist in the left submesh, so it
            # will be mapped to index -1. This is problematic for the
            # assembler, which assumes it is possible to get the full macro
            # dofmap for the trial and test functions, despite the restriction
            # meaning we don't need the non-existant dofs. To fix this, we just
            # map cell_minus to the cell corresponding to cell plus. This will
            # just add zeros to the assembled system, since there are no
            # u("-") terms. Could map this to any cell in the submesh, but
            # I think using the cell on the other side of the facet means a
            # facet space coefficient could be used
            domain_to_domain_0[cell_minus] = domain_to_domain_0[cell_plus]
            # Same hack for the right submesh
            domain_to_domain_1[cell_plus] = domain_to_domain_1[cell_minus]

    return interface_entities, domain_to_domain_0, domain_to_domain_1


def compute_interior_facet_integration_entities(msh, cell_map):
    """
    Compute the integration entities for interior facet integrals.

    Parameters:
        msh: The mesh
        cell_map: A map to apply to the cells in the integration entities

    Returns:
        A (flattened) list of pairs of (cell, local facet index) pairs
    """
    # FIXME Do this more efficiently
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    integration_entities = []
    for facet in range(msh.topology.index_map(fdim).size_local):
        cells = f_to_c.links(facet)
        if len(cells) == 2:
            # FIXME Don't use tolist
            local_facet_plus = c_to_f.links(cells[0]).tolist().index(facet)
            local_facet_minus = c_to_f.links(cells[1]).tolist().index(facet)

            integration_entities.extend(
                [
                    cell_map[cells[0]],
                    local_facet_plus,
                    cell_map[cells[1]],
                    local_facet_minus,
                ]
            )
    return integration_entities


def compute_cell_boundary_facets(msh):
    """Compute the integration entities for integrals around the
    boundaries of all cells in msh.

    Parameters:
        msh: The mesh.

    Returns:
        Facets to integrate over, identified by ``(cell, local facet
        index)`` pairs.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    n_f = cell_num_entities(msh.topology.cell_type, fdim)
    n_c = msh.topology.index_map(tdim).size_local
    return np.vstack(
        (np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))
    ).T.flatten()
