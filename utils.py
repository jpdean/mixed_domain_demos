import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx import mesh


def reorder_mesh(msh):
    # FIXME Check this is correct
    # FIXME For a high-order mesh, the geom has more dofs so need to modify
    # this
    # FIXME What about quads / hexes?
    tdim = msh.topology.dim
    num_cell_vertices = cell_num_entities(msh.topology.cell_type, 0)
    c_to_v = msh.topology.connectivity(tdim, 0)
    geom_dofmap = msh.geometry.dofmap
    vertex_imap = msh.topology.index_map(0)
    geom_imap = msh.geometry.index_map()
    for i in range(0, len(c_to_v.array), num_cell_vertices):
        topo_perm = np.argsort(vertex_imap.local_to_global(
            c_to_v.array[i:i+num_cell_vertices]))
        geom_perm = np.argsort(geom_imap.local_to_global(
            geom_dofmap.array[i:i+num_cell_vertices]))

        c_to_v.array[i:i+num_cell_vertices] = \
            c_to_v.array[i:i+num_cell_vertices][topo_perm]
        geom_dofmap.array[i:i+num_cell_vertices] = \
            geom_dofmap.array[i:i+num_cell_vertices][geom_perm]


def norm_L2(comm, v):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(ufl.inner(v, v) * ufl.dx)), op=MPI.SUM))


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            fem.Constant(msh, PETSc.ScalarType(1.0)) * ufl.dx)), op=MPI.SUM)
    return 1 / vol * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * ufl.dx)), op=MPI.SUM)


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
                assert local_facet >= 0 and local_facet <= 2
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
