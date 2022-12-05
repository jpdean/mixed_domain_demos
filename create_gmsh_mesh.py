import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.mesh import create_cell_partitioner, GhostMode


def create():
    comm = MPI.COMM_WORLD
    gdim = 2

    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("unit_square")
        lc = 1e-1
        gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
        gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
        gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
        gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 1, 4)

        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)

        gmsh.model.geo.addPlaneSurface([1], 1)

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [1], 1)

        gmsh.model.addPhysicalGroup(1, [1], 1)
        gmsh.model.addPhysicalGroup(1, [2], 2)
        gmsh.model.addPhysicalGroup(1, [3], 3)
        gmsh.model.addPhysicalGroup(1, [4], 4)

        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.model.mesh.generate(2)

        # gmsh.write("t1.msh")

    partitioner = create_cell_partitioner(GhostMode.none)
    msh, _, ft = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
    gmsh.finalize()

    return msh, ft


if __name__ == "__main__":
    msh, ft = create()

    from dolfinx import io
    with io.XDMFFile(msh.comm, "mesh.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_meshtags(ft)
