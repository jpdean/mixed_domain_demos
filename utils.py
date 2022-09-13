import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI
from dolfinx.cpp.mesh import cell_num_entities


def reorder_mesh(msh):
    # FIXME Check this is correct
    # FIXME For a high-order mesh, the geom has more dofs so need to modify this
    # FIXME What about quads / hexes?
    tdim = msh.topology.dim
    num_cell_vertices = cell_num_entities(msh.topology.cell_type, 0)
    c_to_v = msh.topology.connectivity(tdim, 0)
    geom_dofmap = msh.geometry.dofmap
    vertex_imap = msh.topology.index_map(0)
    geom_imap = msh.geometry.index_map()
    for i in range(0, len(c_to_v.array), num_cell_vertices):
        topo_perm = np.argsort(vertex_imap.local_to_global(c_to_v.array[i:i+num_cell_vertices]))
        geom_perm = np.argsort(geom_imap.local_to_global(geom_dofmap.array[i:i+num_cell_vertices]))

        c_to_v.array[i:i+num_cell_vertices] = c_to_v.array[i:i+num_cell_vertices][topo_perm]
        geom_dofmap.array[i:i+num_cell_vertices] = geom_dofmap.array[i:i+num_cell_vertices][geom_perm]


def norm_L2(comm, v):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(ufl.inner(v, v) * ufl.dx)), op=MPI.SUM))
