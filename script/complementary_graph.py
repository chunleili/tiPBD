# Change edge to vertex, and vertex to edge, finally generate the mesh(obj or ply). This is complementary graph.
import pyamg
import pyamg.vis
from pyamg.vis.vtk_writer import write_basic_mesh, write_vtu
import numpy as np
from scipy.io import  mmread
import meshio
import taichi as ti
import pathlib


case_name = "scale3"

outdir = f'result/{case_name}/vis/'
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
readdir = f'result/{case_name}/obj/'

ti.init()
N=3
NV = (N + 1)**2
NE = 2 * N * (N + 1) + N**2
edge        = ti.Vector.field(2, dtype=int, shape=(NE))
edge_center = ti.Vector.field(3, dtype=float, shape=(NE))
pos         = ti.Vector.field(3, dtype=float, shape=(NV))
v2edge      = ti.field(dtype=int, shape=(NV,8)) # one vertex may belong to 8 edges at most
num_v2edge  = ti.field(dtype=int, shape=(NV)) # number of edges that a vertex belongs to
v2edge.fill(-1)
adjacent_edge = ti.field(dtype=int, shape=(NE, 20))
num_adjacent_edge = ti.field(dtype=int, shape=(NE))


@ti.kernel
def init_edge(
    edge:ti.template(),
):
    for i, j in ti.ndrange(N + 1, N):
        edge_idx = i * N + j
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        edge_idx = start + j * N + i
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
    start = 2 * N * (N + 1)
    for i, j in ti.ndrange(N, N):
        edge_idx = start + i * N + j
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
        else:
            edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

@ti.kernel
def init_pos(
    pos:ti.template(),
):
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        pos[idx] = ti.Vector([i / N, 0.5, j / N]) # horizontal hang


@ti.kernel
def init_edge_center(
    edge_center:ti.template(),
    edge:ti.template(),
    pos:ti.template(),
):
    for i in range(NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        edge_center[i] = (p1 + p2) / 2.0


# v2e: given vertex idx, return the edge index it belongs to
@ti.kernel
def calc_v2edge(
    edge:ti.template(),
    v2edge:ti.template(),
):
    ti.loop_config(serialize=True)
    for i in range(NE):
        idx1, idx2 = edge[i]

        v2edge[idx1, num_v2edge[idx1]] = i
        num_v2edge[idx1] += 1
        
        v2edge[idx2, num_v2edge[idx2]] = i
        num_v2edge[idx2] += 1
        

@ti.kernel
def init_adjacent_edge_kernel(adjacent_edge:ti.template(),
                            num_adjacent_edge:ti.template(),
                            edge:ti.template()):
    for i in range(NE):
        for j in range(adjacent_edge.shape[1]):
            adjacent_edge[i,j] = -1

    ti.loop_config(serialize=True)
    for i in range(NE):
        a=edge[i][0]
        b=edge[i][1]
        for j in range(i+1, NE):
            if j==i:
                continue
            a1=edge[j][0]
            b1=edge[j][1]
            if a==a1 or a==b1 or b==a1 or b==b1:
                numi = num_adjacent_edge[i]
                numj = num_adjacent_edge[j]
                adjacent_edge[i,numi]=j
                adjacent_edge[j,numj]=i
                num_adjacent_edge[i]+=1
                num_adjacent_edge[j]+=1 



def output_graph(edge_center, adjacent_edge, frame=1):
    points = edge_center
    fname = outdir+f"comp_graph_{frame}.vtu"
    # cells = {1: adjacent_edge}
    # write_vtu(V=V, cells=cells, fname=outdir+f"comp_graph_{frame}.vtu")
    # meshio.write(outdir+f"comp_graph_{frame}.obj", meshio.Mesh(V, {"line": adjacent_edge}))

    cells = [("line", adjacent_edge,)]
    meshio.write_points_cells(fname, points, cells)







if __name__ == '__main__':
    init_edge(edge)
    calc_v2edge(edge, v2edge)
    init_adjacent_edge_kernel(adjacent_edge, num_adjacent_edge, edge)
    init_edge_center(edge_center, edge, pos)
    edge = edge.to_numpy()
    v2edge = v2edge.to_numpy()
    num_v2edge = num_v2edge.to_numpy()
    adjacent_edge = adjacent_edge.to_numpy()
    num_adjacent_edge = num_adjacent_edge.to_numpy()
    edge_center = edge_center.to_numpy()
    output_graph(edge_center, adjacent_edge, 1)