import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh
import sys


def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def norm_L2(comm, v, measure=ufl.dx):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(ufl.inner(v, v) * measure)), op=MPI.SUM))


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            fem.Constant(msh, PETSc.ScalarType(1.0)) * ufl.dx)), op=MPI.SUM)
    return 1 / vol * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * ufl.dx)), op=MPI.SUM)


def normal_jump_error(msh, v):
    n = ufl.FacetNormal(msh)
    return norm_L2(msh.comm, ufl.jump(v, n), measure=ufl.dS)


# FIXME This should be a C++ helper function
# TODO Simplify and document
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
                submesh_cell = cell_map.index(cell)
                submesh_facet = submesh_c_to_f.links(submesh_cell)[local_facet]
                submesh_facets.append(submesh_facet)
                submesh_values.append(facet_tag.values[i])
    submesh_facets = np.array(submesh_facets)
    submesh_values = np.array(submesh_values, dtype=np.intc)
    # Sort and make unique
    submesh_facets, ind = np.unique(submesh_facets, return_index=True)
    submesh_values = submesh_values[ind]
    submesh_meshtags = mesh.meshtags(
        submesh, submesh.topology.dim - 1, submesh_facets, submesh_values)
    return submesh_meshtags


def create_random_mesh(corners, n, ghost_mode):
    """Create a rectangular mesh made of randomly ordered simplices"""
    if MPI.COMM_WORLD.rank == 0:
        h_x = (corners[1][0] - corners[0][0]) / n[0]
        h_y = (corners[1][1] - corners[0][1]) / n[1]

        points = [(i * h_x, j * h_y)
                  for i in range(n[0] + 1) for j in range(n[1] + 1)]

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
    domain = ufl.Mesh(basix.ufl_wrapper.create_vector_element(
        "Lagrange", "triangle", 1))
    partitioner = mesh.create_cell_partitioner(ghost_mode)
    return mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain,
                            partitioner=partitioner)


def create_trap_mesh(comm, n, corners, offset_scale=0.25,
                     ghost_mode=mesh.GhostMode.none):
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
                        offset = - offset_scale * h[1]
                x.append([corners[0][0] + i * h[0],
                          corners[0][1] + j * h[1] + offset])
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


class TimeDependentExpression():
    """Simple class to represent time dependent functions"""

    def __init__(self, expression):
        self.t = 0
        self.expression = expression

    def __call__(self, x):
        return self.expression(x, self.t)
