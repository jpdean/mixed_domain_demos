import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh
import sys
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.cpp.fem import compute_integration_domains


def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def norm_L2(comm, v, measure=ufl.dx):
    return np.sqrt(
        comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(v, v) * measure)), op=MPI.SUM)
    )


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, PETSc.ScalarType(1.0)) * ufl.dx)),
        op=MPI.SUM,
    )
    return 1 / vol * msh.comm.allreduce(fem.assemble_scalar(fem.form(v * ufl.dx)), op=MPI.SUM)


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

    domain = ufl.Mesh(basix.ufl_wrapper.create_vector_element("Lagrange", "triangle", 1))
    partitioner = mesh.create_cell_partitioner(ghost_mode)
    return mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain, partitioner=partitioner)


def create_trap_mesh(comm, n, corners, offset_scale=0.25, ghost_mode=mesh.GhostMode.none):
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


def interface_int_entities(msh, interface_facets, marker):
    """
    This helper function computes the integration entities for interior facet
    integrals (i.e. a list of (cell_0, local_facet_0, cell_1, local_facet_1))
    over an interface. The integration entities are ordered consistently such
    that cells for which `marker[cell] != 0` correspond to the "+" restriction,
    and cells for which `marker[cell] == 0` correspond to the "-" restriction.

    Parameters:
        msh: the mesh
        interface_facets: Facet indices of interior facets on an interface
        marker: If `marker[cell] != 0`, then that cell corresponds to a "+"
            restriction. Otherwise, it corresponds to a negative restriction.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    facet_imap = msh.topology.index_map(fdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    interface_entities = []
    for facet in interface_facets:
        # Check if this facet is owned
        if facet < facet_imap.size_local:
            cells = f_to_c.links(facet)
            assert len(cells) == 2
            if marker[cells[0]] == 0:
                cell_plus, cell_minus = cells[1], cells[0]
            else:
                cell_plus, cell_minus = cells[0], cells[1]

            local_facet_plus = np.where(c_to_f.links(cell_plus) == facet)[0][0]
            local_facet_minus = np.where(c_to_f.links(cell_minus) == facet)[0][0]

            interface_entities.extend([cell_plus, local_facet_plus, cell_minus, local_facet_minus])

    return interface_entities


def interior_facet_int_entities(msh, cell_map):
    """
    Compute the integration entities for interior facet integrals.

    Parameters:
        msh: The mesh
        cell_map: An EntityMap for the cells in msh where msh is the
            sub-topology and the EntityMap's topology is the target
            topology to map to

    Returns:
        A (flattened) list of pairs of (cell, local facet index) pairs
    """

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, tdim)

    facets = np.arange(msh.topology.index_map(fdim).size_local, dtype=np.int32)
    integration_entities = compute_integration_domains(
        fem.IntegralType.interior_facet, msh.topology._cpp_object, facets
    )

    cells = integration_entities[::2]
    mapped_cells = cell_map.sub_topology_to_topology(cells, inverse=False)
    integration_entities[::2] = mapped_cells
    return integration_entities


def compute_cell_boundary_int_entities(msh):
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
    return np.vstack((np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))).T.flatten()


# TODO Optimise
def one_sided_int_entities(msh, facets):
    """
    Give a list of facets, this function returns the corresponding
    (cell, local facet index) pairs. For facets with two connected cells,
    the first cell in the facet to cell connectivity is taken.

    Parameters:
        msh: The mesh
        facets: The facets

    Returns:
        A list of (cell, local facet index) pairs corresponding to the input facets
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    facet_imap = msh.topology.index_map(fdim)
    facet_integration_entities = []
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    # Loop through all interface facets
    for facet in facets:
        # Check if this facet is owned
        if facet < facet_imap.size_local:
            # Get a cell connected to the facet and extract the local facet index
            cell = f_to_c.links(facet)[0]
            local_facet = np.where(c_to_f.links(cell) == facet)[0][0]
            facet_integration_entities.extend([cell, local_facet])

    return facet_integration_entities


def markers_to_meshtags(msh, tags, markers, dim):
    """
    Create meshtags from a given set of tags and markers. Tag[i] is is the tag for
    the part of the domain marked by markers[i].\

    Parameters:
        msh: The mesh
        tags: A list of tags
        markers: A list of markers
        dim: The dimension of the entities
    """
    entities = [mesh.locate_entities_boundary(msh, dim, marker) for marker in markers]
    values = [np.full_like(entities, tag) for (tag, entities) in zip(tags, entities)]
    entities = np.hstack(entities, dtype=np.int32)
    values = np.hstack(values, dtype=np.intc)
    perm = np.argsort(entities)
    return mesh.meshtags(msh, dim, entities[perm], values[perm])


def jump_i(v, n):
    return v[0]("+") * n("+") + v[1]("-") * n("-")


def grad_avg_i(v, kappa):
    return kappa[1] / (kappa[0] + kappa[1]) * kappa[0] * ufl.grad(v[0]("+")) + kappa[0] / (
        kappa[0] + kappa[1]
    ) * kappa[1] * ufl.grad(v[1]("-"))
