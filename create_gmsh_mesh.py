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
        lc = 0.1

        # Point tags
        num_bottom_points = 100
        bottom_points = [gmsh.model.geo.addPoint(x, 0, 0, lc)
                         for x in np.linspace(0.0, 3.0, num_bottom_points)]
        top_left_point = gmsh.model.geo.addPoint(0, 1, 0, lc)
        top_right_point = gmsh.model.geo.addPoint(3, 1, 0, lc)

        # Line tags
        lines = []
        lines.append(gmsh.model.geo.addSpline(bottom_points))
        lines.append(gmsh.model.geo.addLine(bottom_points[-1],
                                            top_right_point))
        lines.append(gmsh.model.geo.addLine(top_right_point,
                                            top_left_point))
        lines.append(gmsh.model.geo.addLine(top_left_point,
                                            bottom_points[0]))

        gmsh.model.geo.addCurveLoop(lines, 1)

        gmsh.model.geo.addPlaneSurface([1], 1)

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [1], 1)

        gmsh.model.addPhysicalGroup(1, [lines[0]], 1)
        gmsh.model.addPhysicalGroup(1, [lines[1]], 2)
        gmsh.model.addPhysicalGroup(1, [lines[2]], 3)
        gmsh.model.addPhysicalGroup(1, [lines[3]], 4)

        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
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
