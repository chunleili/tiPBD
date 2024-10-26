import ctypes
import scipy.sparse
import taichi as ti
import numpy as np
import time
import scipy
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
from pathlib import Path
import os,sys
from matplotlib import pyplot as plt
import shutil, glob
import meshio
import tqdm
import argparse
from collections import namedtuple
import json
import logging
import datetime
from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial

from time import perf_counter
import pyamg
import numpy.ctypeslib as ctl

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from engine.file_utils import process_dirs,  do_restart, save_state,  export_A_b
from engine.mesh_io import write_mesh
from engine.solver.build_Ps import build_Ps
from engine.solver.amg_python import AMG_python

prj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"


#parse arguments to change default values
from engine.common_args import add_common_args
parser = argparse.ArgumentParser()
parser = add_common_args(parser)

parser.add_argument("-N", type=int, default=64)
parser.add_argument("-compliance", type=float, default=1.0e-8)
parser.add_argument("-use_PXPBD_v1", type=int, default=False)
parser.add_argument("-use_PXPBD_v2", type=int, default=False)
parser.add_argument("-use_bending", type=int, default=False)
parser.add_argument("-setup_num", type=int, default=0, help="attach:0, scale:1")

parser.add_argument("-omega", type=float, default=0.25)
parser.add_argument("-smoother_type", type=str, default="chebyshev")

args = parser.parse_args()

N = args.N

if args.setup_num==1: args.gravity = (0.0, 0.0, 0.0)
else : args.gravity = (0.0, -9.8, 0.0)

args.save_image = True
args.use_viewer = False
args.use_geometric_stiffness = False
args.export_fullr = False
args.calc_r_xpbd = True
args.use_cpp_initFill = True
args.PXPBD_ksi = 1.0


if args.arch == "gpu":
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)

if args.use_PXPBD_v1 or args.use_PXPBD_v2:
    ResidualDataPrimal = namedtuple('residual', ['dual','primal','Newton', 'ninner','t'])
    Residual0DataPrimal = namedtuple('residual', ['dual','primal','Newton'])
else:
    ResidualData = namedtuple('residual', ['dual', 'ninner','t'])
    Residual0Data = namedtuple('residual', ['dual'])

class Cloth():
    def __init__(self) -> None:
        self.Ps = None
        self.num_levels = 0
        self.paused = False
        self.n_outer_all = [] 
        self.t_export = 0.0
        self.sim_name=f"cloth-N{args.N}"

        self.frame = 0
        self.ite=0
        self.NV = (N + 1)**2
        self.NT = 2 * N**2
        self.NE = 2 * N * (N + 1) + N**2
        self.NCONS = self.NE
        self.new_M = int(self.NE / 100)
        self.compliance = args.compliance  #see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
        self.alpha = self.compliance * (1.0 / args.delta_t / args.delta_t)  # timestep related compliance, see XPBD self.paper
        self.alpha_bending = 1.0 * (1.0 / args.delta_t / args.delta_t) #TODO: need to be tuned
    
        self.ALPHA = None
        
        self.tri = ti.field(ti.i32, shape=3 * self.NT)
        self.edge        = ti.Vector.field(2, dtype=int, shape=(self.NE))
        self.pos         = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.dpos        = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.dpos_withg  = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.old_pos     = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.vel         = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.pos_mid     = ti.Vector.field(3, dtype=float, shape=(self.NV))
        self.inv_mass    = ti.field(dtype=float, shape=(self.NV))
        self.rest_len    = ti.field(dtype=float, shape=(self.NE))
        self.lagrangian  = ti.field(dtype=float, shape=(self.NE))  
        self.constraints = ti.field(dtype=float, shape=(self.NE))  
        self.dLambda     = ti.field(dtype=float, shape=(self.NE))
        # self.# numerator   = ti.field(dtype=float, shape=(self.NE))
        # self.# denominator = ti.field(dtype=float, shape=(self.NE))
        self.gradC       = ti.Vector.field(3, dtype = ti.float32, shape=(self.NE,2)) 
        self.edge_center = ti.Vector.field(3, dtype = ti.float32, shape=(self.NE))
        self.dual_residual       = ti.field(shape=(self.NE),    dtype = ti.float32) # -C - alpha * self.lagrangian
        self.nnz_each_row = np.zeros(self.NE, dtype=int)
        self.potential_energy = ti.field(dtype=float, shape=())
        self.inertial_energy = ti.field(dtype=float, shape=())
        self.predict_pos = ti.Vector.field(3, dtype=float, shape=(self.NV))
        # self.# primary_residual = np.zeros(dtype=float, shape=(3*self.NV))
        # self.# K = ti.Matrix.field(3, 3, float, (self.NV, self.NV)) 
        # self.# geometric stiffness, only retain diagonal elements
        self.K_diag = np.zeros((self.NV*3), dtype=float)
        # self.# Minv_gg = ti.Vector.field(3, dtype=float, shape=(self.NV))



def init_extlib_argtypes():
    global extlib

    # DEBUG only
    # os.chdir(prj_path+'/cpp/mgcg_cuda')
    # os.system("cmake --build build --config Debug")
    # os.chdir(prj_path)

    os.add_dll_directory(args.cuda_dir)
    extlib = ctl.load_library("fastmg.dll", prj_path+'/cpp/mgcg_cuda/lib')

    arr_int = ctl.ndpointer(dtype=np.int32, ndim=1, flags='aligned, c_contiguous')
    arr_float = ctl.ndpointer(dtype=np.float32, ndim=1, flags='aligned, c_contiguous')
    arr2d_float = ctl.ndpointer(dtype=np.float32, ndim=2, flags='aligned, c_contiguous')
    arr2d_int = ctl.ndpointer(dtype=np.int32, ndim=2, flags='aligned, c_contiguous')
    c_size_t = ctypes.c_size_t
    c_float = ctypes.c_float
    c_int = ctypes.c_int
    argtypes_of_csr=[ctl.ndpointer(np.float32,flags='aligned, c_contiguous'),    # data
                    ctl.ndpointer(np.int32,  flags='aligned, c_contiguous'),      # indices
                    ctl.ndpointer(np.int32,  flags='aligned, c_contiguous'),      # indptr
                    ctypes.c_int, ctypes.c_int, ctypes.c_int           # rows, cols, nnz
                    ]

    extlib.fastmg_set_data.argtypes = [arr_float, c_size_t, arr_float, c_size_t, c_float, c_size_t]
    extlib.fastmg_get_data.argtypes = [arr_float]*2
    extlib.fastmg_get_data.restype = c_size_t
    extlib.fastmg_setup_nl.argtypes = [ctypes.c_size_t]
    extlib.fastmg_RAP.argtypes = [ctypes.c_size_t]
    extlib.fastmg_set_A0.argtypes = argtypes_of_csr
    extlib.fastmg_set_P.argtypes = [ctypes.c_size_t] + argtypes_of_csr
    extlib.fastmg_setup_smoothers.argtypes = [c_int]
    extlib.fastmg_update_A0.argtypes = [arr_float]
    extlib.fastmg_get_data.restype = c_int

    extlib.fastFillCloth_set_data.argtypes = [arr2d_int, c_int, arr_float, c_int, arr2d_float, c_float]
    extlib.fastFillCloth_run.argtypes = [arr2d_float]
    extlib.fastFillCloth_fetch_A_data.argtypes = [arr_float]
    extlib.fastFillCloth_init_from_python_cache.argtypes = [arr2d_int, arr_int, arr2d_int, c_int, arr_float, arr_int, arr_int, arr_int, arr_int, c_int, c_int]

    extlib.initFillCloth_set.argtypes = [arr2d_int, c_int]
    extlib.initFillCloth_get.argtypes = [arr2d_int, arr_int, arr2d_int, c_int] + [arr_int]*4 + [arr2d_int, arr_int]

    extlib.initFillCloth_new()

    extlib.fastmg_new()

    extlib.fastFillCloth_new()

if args.use_cuda:
    init_extlib_argtypes()



@ti.kernel
def init_pos(
    inv_mass:ti.template(),
    pos:ti.template(),
    N:ti.i32,
    NV:ti.i32,
):
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        # pos[idx] = ti.Vector([i / N,  j / N, 0.5])  # vertical hang
        pos[idx] = ti.Vector([i / N, 0.5, j / N]) # horizontal hang
        inv_mass[idx] = 1.0
    if args.setup_num == 0:
        inv_mass[N] = 0.0
        inv_mass[NV-1] = 0.0


@ti.kernel
def init_tri(tri:ti.template()):
    for i, j in ti.ndrange(N, N):
        tri_idx = 6 * (i * N + j)
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 2
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2
        else:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 1
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx + 1
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2


@ti.kernel
def init_edge(
    edge:ti.template(),
    rest_len:ti.template(),
    pos:ti.template(),
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
    for i in range(ist.NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        rest_len[i] = (p1 - p2).norm()

@ti.kernel
def init_edge_center(
    edge_center:ti.template(),
    edge:ti.template(),
    pos:ti.template(),
):
    for i in range(ist.NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        edge_center[i] = (p1 + p2) / 2.0



# tri:  (num_tri, 3)
# https://matthias-research.github.io/pages/tenMinutePhysics/14-cloth.pdf last page
# https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/14-cloth.html
def find_tri_neighbors(tri):
    # 1. Build the edge list
    num_tri = tri.shape[0]
    # This is different with existing edge. 
    # This new edge list may have duplicates because in different triangles.
    edge_list = [] # [v0, v1, gid]
    # gid = 3 * tid + 0/1/2(local id)
    for t in range(num_tri):
        v0 = tri[t, 0]
        v1 = tri[t, 1]
        v2 = tri[t, 2]
        # [0,1][1,2][2,0]: the counter-clockwise order, we use this
        # [0,2][2,1][1,0]: the clockwise order, also reasonable
        edge_list.append([min(v0, v1), max(v0, v1), 3 * t + 0])
        edge_list.append([min(v1, v2), max(v1, v2), 3 * t + 1])
        edge_list.append([min(v2, v0), max(v2, v0), 3 * t + 2])

    # 2. Sort the edge list by the two vertices
    edge_list = np.array(edge_list)
    sorted_indices = np.lexsort((edge_list[:, 1], edge_list[:, 0]))
    sorted = edge_list[sorted_indices]


    # 3. Find the tri neighbors by duplicated edges
    # tri_neighbor: (num_tri, 3), each row is the gid of the neighbor triangle. We could also use tid instead of gid, but gid gives more information. gid//3 is the tid(neigbour triangle), gid%3 is the local edge id(which edge of the  neighbor triangle)
    tri_neighbor = np.ones((num_tri, 3), dtype=np.int32) * (-1)
    for i in range(0, len(sorted), 2):
        # If the first 2 values of sorted edge list is the same, then  they are the same edge with different triangles. Then these two triangles are neighbors.
        if i + 1 < len(sorted) and \
        (sorted[i][0] == sorted[i + 1][0]) and \
        (sorted[i][1] == sorted[i + 1][1]):
            gid0 = sorted[i, 2]
            gid1 = sorted[i + 1, 2]
            tid0 = gid0 // 3     # triangle id
            tid1 = gid1 // 3 
            localid0 = gid0 % 3  # which edge(0/1/2)
            localid1 = gid1 % 3
            # CAUTION: We store gid instead of tid
            tri_neighbor[tid0, localid0] = gid1  
            tri_neighbor[tid1, localid1] = gid0
    return tri_neighbor


def build_tri_pairs(tri, tri_neighbor):
    num_tri = tri.shape[0]
    tri_pairs = []
    for i in range(num_tri): # 遍历三角形
        for j in range(3):   # 遍历三角形的三个边
            gid = tri_neighbor[i, j] #三角形i的第j条边的邻居global edge id，如果没有邻居则为-1。 
            # gid = 3 * tri_id + 0/1/2 
            # 因此gid % 3 得到local edge id, 即共享边是邻居三角形的第几个边
            # gid // 3 得到tri_id，即邻居三角形的id
            if gid >= 0: # 有邻居, 即不是-1
                tid = gid // 3      # 邻居三角形的id
                localid = gid % 3   # 邻居三角形的第几个边
                id0 = tri[i, j]             # 三角形i的第0个点
                id1 = tri[i, (j + 1) % 3]   # 三角形i的第1个点
                id2 = tri[i, (j + 2) % 3]   # 三角形i的第2个点
                id3 = tri[tid, (localid + 2) % 3] # 邻居三角形的非共享点, 即它的第三个点
                tri_pairs.append([id0, id1, id2, id3])
    return tri_pairs


def init_bending_length(tri_pairs, pos):
    bending_id = tri_pairs.copy()
    bending_length = np.zeros(len(bending_id), dtype=np.float32)
    for i in range(bending_length.shape[0]):
        v2 = bending_id[i, 2]
        v3 = bending_id[i, 3]
        bending_length[i] = np.linalg.norm(pos[v2] - pos[v3])
    return bending_length


@ti.kernel
def init_bending_length_kernel(tri_pairs:ti.types.ndarray(), pos:ti.template(), bending_length:ti.types.ndarray()):
    for i in range((tri_pairs).shape[0]):
        v2 = tri_pairs[i, 2]
        v3 = tri_pairs[i, 3]
        bending_length[i] = (pos[v2] - pos[v3]).norm()


def init_bending(tri, pos):
    print("init_bending...")
    tic = perf_counter()
    tic1 = perf_counter()
    tri_neighbor = find_tri_neighbors(tri)
    # print("邻居边编号列表:", tri_neighbor)
    print(f"find_tri_neighbors time: {perf_counter() - tic1}")

    # tri_pairs有四个点，第四个点是另一个三角形的点
    tic2 = perf_counter()
    tri_pairs = build_tri_pairs(tri, tri_neighbor)
    tri_pairs = np.array(tri_pairs, dtype=np.int32)
    # print("三角形对列表:", tri_pairs)
    print(f"build_tri_pairs time: {perf_counter() - tic2}")

    tic3 = perf_counter()
    # bending_length = init_bending_length(tri_pairs, pos.to_numpy())
    bending_length = np.zeros(len(tri_pairs), dtype=np.float32)
    init_bending_length_kernel(tri_pairs, pos, bending_length)
    # print("弯曲长度列表:", bending_length)
    print(f"init_bending_length time: {perf_counter() - tic3}")
    print(f"init_bending time: {perf_counter() - tic}")
    return tri_pairs, bending_length







def read_tri_cloth(filename):
    edge_file_name = filename + ".edge"
    node_file_name = filename + ".node"
    face_file_name = filename + ".face"

    with open(node_file_name, "r") as f:
        lines = f.readlines()
        ist.NV = int(lines[0].split()[0])
        pos = np.zeros((ist.NV, 3), dtype=np.float32)
        for i in range(ist.NV):
            pos[i] = np.array(lines[i + 1].split()[1:], dtype=np.float32)

    with open(edge_file_name, "r") as f:
        lines = f.readlines()
        ist.NE = int(lines[0].split()[0])
        edge_indices = np.zeros((ist.NE, 2), dtype=np.int32)
        for i in range(ist.NE):
            edge_indices[i] = np.array(lines[i + 1].split()[1:], dtype=np.int32)

    with open(face_file_name, "r") as f:
        lines = f.readlines()
        NF = int(lines[0].split()[0])
        face_indices = np.zeros((NF, 3), dtype=np.int32)
        for i in range(NF):
            face_indices[i] = np.array(lines[i + 1].split()[1:-1], dtype=np.int32)

    return pos, edge_indices, face_indices.flatten()


def read_tri_cloth_obj(path):
    print(f"path is {path}")
    mesh = meshio.read(path)
    tri = mesh.cells_dict["triangle"]
    pos = mesh.points

    num_tri = len(tri)
    edges=[]
    for i in range(num_tri):
        ele = tri[i]
        edges.append([min((ele[0]), (ele[1])), max((ele[0]),(ele[1]))])
        edges.append([min((ele[1]), (ele[2])), max((ele[1]),(ele[2]))])
        edges.append([min((ele[0]), (ele[2])), max((ele[0]),(ele[2]))])
    #remove the duplicate edges
    # https://stackoverflow.com/questions/2213923/removing-duplicates-from-a-list-of-lists
    import itertools
    edges.sort()
    edges = list(edges for edges,_ in itertools.groupby(edges))

    return pos, np.array(edges), tri.flatten()


@ti.kernel
def semi_euler(
    old_pos:ti.template(),
    inv_mass:ti.template(),
    vel:ti.template(),
    pos:ti.template(),
    predict_pos:ti.template(),
    delta_t:ti.f32,
):
    g = ti.Vector(args.gravity)
    for i in range(ist.NV):
        if inv_mass[i] != 0.0:
            vel[i] += delta_t * g
            old_pos[i] = pos[i]
            pos[i] += delta_t * vel[i]
            predict_pos[i] = pos[i]


@ti.kernel
def solve_constraints(
    inv_mass:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
):
    for i in range(ist.NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        if invM0 != 0.0:
            dpos[idx0] += invM0 * l * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * l * gradient



@ti.kernel
def solve_distance_constraints_xpbd(
    dual_residual: ti.template(),
    inv_mass:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
):
    for i in range(ist.NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        delta_lagrangian = -(constraint + lagrangian[i] * ist.alpha) / (invM0 + invM1 + ist.alpha)
        lagrangian[i] += delta_lagrangian

        # residual
        dual_residual[i] = -(constraint + ist.alpha * lagrangian[i])
        
        if invM0 != 0.0:
            dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient


@ti.kernel
def solve_bending_constraints_xpbd(
    dual_residual_bending: ti.template(),
    inv_mass:ti.template(),
    lagrangian_bending:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    bending_length:ti.types.ndarray(),
    tri_pairs:ti.types.ndarray(),
):
    for i in range(bending_length.shape[0]):
        idx0, idx1 = tri_pairs[i, 2], tri_pairs[i, 3]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        if invM0 == 0.0 and invM1 == 0.0:
            continue
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - bending_length[i]
        gradient = dis.normalized()
        if gradient.norm() == 0.0:
            continue
        l = -constraint / (invM0 + invM1)
        delta_lagrangian = -(constraint + lagrangian_bending[i] * args.alpha_bending) / (invM0 + invM1 + args.alpha_bending)
        lagrangian_bending[i] += delta_lagrangian

        # residual
        dual_residual_bending[i] = (constraint + args.alpha_bending * lagrangian_bending[i])
        
        if invM0 != 0.0:
            dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient


@ti.kernel
def update_pos(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    omega:ti.f32,
):
    for i in range(ist.NV):
        if inv_mass[i] != 0.0:
            pos[i] += omega * dpos[i]


@ti.kernel
def update_pos_blend(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    dpos_withg:ti.template(),
):
    for i in range(ist.NV):
        if inv_mass[i] != 0.0:
            pos[i] += args.omega *((1-args.PXPBD_ksi) * dpos[i] + args.PXPBD_ksi * dpos_withg[i])


@ti.kernel
def update_vel(
    old_pos:ti.template(),
    inv_mass:ti.template(),    
    vel:ti.template(),
    pos:ti.template(),
):
    for i in range(ist.NV):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / args.delta_t


@ti.kernel 
def reset_dpos(dpos:ti.template()):
    for i in range(ist.NV):
        dpos[i] = ti.Vector([0.0, 0.0, 0.0])



@ti.kernel
def calc_dual_residual(
    dual_residual: ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    pos:ti.template(),
):
    for i in range(ist.NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]

        # residual(lagrangian=0 for first iteration)
        dual_residual[i] = -(constraint + ist.alpha * lagrangian[i])

def calc_primary_residual(G,M_inv):
    MASS = sp.diags(1.0/(M_inv.diagonal()+1e-12), format="csr")
    primary_residual = MASS @ (ist.pos.to_numpy().flatten() - ist.predict_pos.to_numpy().flatten()) - G.transpose() @ ist.lagrangian.to_numpy()
    where_zeros = np.where(M_inv.diagonal()==0)
    primary_residual = np.delete(primary_residual, where_zeros)
    return primary_residual


def xpbd_calcr(tic_iter, dual0, r):
    tic_calcr = perf_counter()
    t_iter = perf_counter()-tic_iter
    # dualr = np.linalg.norm(dual_residual.to_numpy())
    dualr = calc_norm(ist.dual_residual)
    
    # if export_fullr:
    #     np.savez(args.out_dir+'/r/'+ f'fulldual_{ist.frame}-{ist.ite}', fulldual0)

    t_calcr = perf_counter()-tic_calcr
    tic_exportr = perf_counter()
    r.append(ResidualData(dualr, 1, t_iter))
    if args.export_log:
        logging.info(f"{ist.frame}-{ist.ite}  dual0:{dual0:.2e} dual:{dualr:.2e}  t:{t_iter:.2e}s calcr:{t_calcr:.2e}s")
    ist.t_export += perf_counter() - tic_exportr
    return dualr, dual0


@ti.kernel
def calc_norm(a:ti.template())->ti.f32:
    sum = 0.0
    for i in range(a.shape[0]):
        sum += a[i] * a[i]
    sum = ti.sqrt(sum)
    return sum


all_stalled = []
# if in last 5 iters, residuals not change 0.1%, then it is stalled
def is_stall(r):
    if (ist.ite < 5):
        return False
    # a=np.array([r[-1].dual, r[-2].dual,r[-3].dual,r[-4].dual,r[-5].dual])
    inc1 = r[-1].dual/r[-2].dual
    inc2 = r[-2].dual/r[-3].dual
    inc3 = r[-3].dual/r[-4].dual
    inc4 = r[-4].dual/r[-5].dual
    if args.use_PXPBD_v1:
        inc1 = r[-1].Newton/r[-2].Newton
        inc2 = r[-2].Newton/r[-3].Newton
        inc3 = r[-3].Newton/r[-4].Newton
        inc4 = r[-4].Newton/r[-5].Newton
    
    # if all incs is in [0.999,1.001]
    if np.all((inc1>0.999) & (inc1<1.001) & (inc2>0.999) & (inc2<1.001) & (inc3>0.999) & (inc3<1.001) & (inc4>0.999) & (inc4<1.001)):
        logging.info(f"Stall at {ist.frame}-{ist.ite}")
        all_stalled.append((ist.frame, ist.ite))
        return True
    return False

def substep_xpbd():
    semi_euler(ist.old_pos, ist.inv_mass, ist.vel, ist.pos, ist.predict_pos,args.delta_t)
    reset_lagrangian(ist.lagrangian)

    calc_dual_residual(ist.dual_residual, ist.edge, ist.rest_len, ist.lagrangian, ist.pos)
    fulldual0 = ist.dual_residual.to_numpy()
    dual0 = np.linalg.norm(fulldual0).astype(float)
    r = []
    for ist.ite in range(args.maxiter):
        tic_iter = perf_counter()

        reset_dpos(ist.dpos)
        if args.use_bending:
            # TODO: should use seperate dual_residual_bending and lagrangian_bending
            solve_bending_constraints_xpbd(ist.dual_residual, ist.inv_mass, ist.lagrangian, ist.dpos, ist.pos, ist.bending_length, ist.tri_pairs)
        solve_distance_constraints_xpbd(ist.dual_residual, ist.inv_mass, ist.edge, ist.rest_len, ist.lagrangian, ist.dpos, ist.pos)
        update_pos(ist.inv_mass, ist.dpos, ist.pos,args.omega)

        if args.calc_r_xpbd:
            dualr, dualr0 = xpbd_calcr(tic_iter, dual0, r)

        if dualr<args.tol:
            break
        # if is_stall(r):
        #     logging.info("Stall detected, break")
        #     break
    ist.n_outer_all.append(ist.ite+1)

    if args.export_residual:
        do_export_r(r)
    update_vel(ist.old_pos, ist.inv_mass, ist.vel, ist.pos)




# ---------------------------------------------------------------------------- #
#                                   for ours                                   #
# ---------------------------------------------------------------------------- #
@ti.kernel
def compute_C_and_gradC_kernel(
    pos:ti.template(),
    gradC: ti.template(),
    edge:ti.template(),
    constraints:ti.template(),
    rest_len:ti.template(),
):
    for i in range(ist.NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraints[i] = dis.norm() - rest_len[i]
        g = dis.normalized()

        gradC[i, 0] = g
        gradC[i, 1] = -g


@ti.kernel
def compute_K_kernel(K_diag:ti.types.ndarray(),):
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        dis = ist.pos[idx0] - ist.pos[idx1]
        L= dis.norm()
        g = dis.normalized()

        #geometric stiffness K: 
        # https://github.com/FantasyVR/magicMirror/blob/a1e56f79504afab8003c6dbccb7cd3c024062dd9/geometric_stiffness/meshComparison/meshgs_SchurComplement.py#L143
        # https://team.inria.fr/imagine/files/2015/05/final.pdf eq.21
        # https://blog.csdn.net/weixin_43940314/article/details/139448858
        k0 = ist.lagrangian[i] / L * (1 - g[0]*g[0])
        k1 = ist.lagrangian[i] / L * (1 - g[1]*g[1])
        k2 = ist.lagrangian[i] / L * (1 - g[2]*g[2])
        K_diag[idx0*3]   += k0
        K_diag[idx0*3+1] += k1
        K_diag[idx0*3+2] += k2
        K_diag[idx1*3]   += k0
        K_diag[idx1*3+1] += k1
        K_diag[idx1*3+2] += k2
    ...


@ti.kernel
def update_constraints_kernel(
    pos:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    constraints:ti.template(),
):
    for i in range(ist.NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraints[i] = dis.norm() - rest_len[i]


@ti.kernel
def fill_gradC_triplets_kernel(
    ii:ti.types.ndarray(dtype=ti.i32),
    jj:ti.types.ndarray(dtype=ti.i32),
    vv:ti.types.ndarray(dtype=ti.f32),
    gradC: ti.template(),
    edge: ti.template(),
):
    cnt=0
    ti.loop_config(serialize=True)
    for j in range(edge.shape[0]):
        ind = edge[j]
        for p in range(2):
            for d in range(3):
                pid = ind[p]
                ii[cnt],jj[cnt],vv[cnt] = j, 3 * pid + d, gradC[j, p][d]
                cnt+=1



@ti.kernel
def fill_gradC_np_kernel(
    G: ti.types.ndarray(),
    gradC: ti.template(),
    edge: ti.template(),
):
    for j in edge:
        ind = edge[j]
        for p in range(2): #which point in the edge
            for d in range(3): #which dimension
                pid = ind[p]
                G[j, 3 * pid + d] = gradC[j, p][d]


@ti.kernel
def reset_lagrangian(lagrangian: ti.template()):
    for i in range(ist.NE):
        lagrangian[i] = 0.0


def amg_core_gauss_seidel(Ap, Aj, Ax, x, b, row_start: int, row_stop: int, row_step: int):
    for i in range(row_start, row_stop, row_step):
        start = Ap[i]
        end = Ap[i + 1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = Aj[jj]
            if i == j:
                diag = Ax[jj]
            else:
                rsum += Ax[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag

@ti.kernel
def amg_core_gauss_seidel_kernel(Ap: ti.types.ndarray(),
                                 Aj: ti.types.ndarray(),
                                 Ax: ti.types.ndarray(),
                                 x: ti.types.ndarray(),
                                 b: ti.types.ndarray(),
                                 row_start: int,
                                 row_stop: int,
                                 row_step: int):
    # if row_step < 0:
    #     assert "row_step must be positive"
    for i in range(row_start, row_stop):
        if i%row_step != 0:
            continue

        start = Ap[i]
        end = Ap[i + 1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = Aj[jj]
            if i == j:
                diag = Ax[jj]
            else:
                rsum += Ax[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag


# ---------------------------------------------------------------------------- #
#                                    amgpcg                                    #
# ---------------------------------------------------------------------------- #



def cuda_set_A0(A0):
    extlib.fastmg_set_A0(A0.data.astype(np.float32), A0.indices, A0.indptr, A0.shape[0], A0.shape[1], A0.nnz)


def cuda_update_A0(A0):
    extlib.fastmg_update_A0(A0.data.astype(np.float32))


def AMG_solve(b, x0=None, tol=1e-5, maxiter=100):
    if x0 is None:
        x0 = np.zeros(b.shape[0], dtype=np.float32)

    tic4 = time.perf_counter()
    # set data
    x0 = x0.astype(np.float32)
    b = b.astype(np.float32)
    extlib.fastmg_set_data(x0, x0.shape[0], b, b.shape[0], tol, maxiter)

    # solve
    extlib.fastmg_solve()

    # get result
    x = np.empty_like(x0, dtype=np.float32)
    residuals = np.zeros(shape=(maxiter,), dtype=np.float32)
    niter = extlib.fastmg_get_data(x, residuals)
    niter += 1
    residuals = residuals[:niter]
    logging.info(f"    inner iter: {niter}")
    logging.info(f"    solve time: {(time.perf_counter()-tic4)*1000:.0f}ms")
    return (x),  residuals  




def update_P(Ps):
    for lv in range(len(Ps)):
        P_ = Ps[lv]
        extlib.fastmg_set_P(lv, P_.data.astype(np.float32), P_.indices, P_.indptr, P_.shape[0], P_.shape[1], P_.nnz)

# ---------------------------------------------------------------------------- #
#                                  amgpcg end                                  #
# ---------------------------------------------------------------------------- #

# @ti.kernel
# def calc_chen2023_added_dpos(G, M_inv, Minv_gg, dLambda):
#     dpos = M_inv @ G.transpose() @ dLambda
#     dpos -= Minv_gg
#     return dpos

def transfer_back_to_pos_matrix(x, M_inv, G, pos_mid, Minv_gg=None):
    dLambda_ = x.copy()
    ist.lagrangian.from_numpy(ist.lagrangian.to_numpy() + dLambda_)
    dpos = M_inv @ G.transpose() @ dLambda_ 
    if args.use_PXPBD_v1:
        dpos -=  Minv_gg
    dpos = dpos.reshape(-1, 3)
    ist.pos.from_numpy(pos_mid.to_numpy() + args.omega*dpos)

@ti.kernel
def transfer_back_to_pos_mfree_kernel():
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = ist.inv_mass[idx0], ist.inv_mass[idx1]

        delta_lagrangian = ist.dLambda[i]
        ist.lagrangian[i] += delta_lagrangian

        gradient = ist.gradC[i, 0]
        
        if invM0 != 0.0:
            ist.dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            ist.dpos[idx1] -= invM1 * delta_lagrangian * gradient


@ti.kernel
def transfer_back_to_pos_mfree_kernel_withg():
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = ist.inv_mass[idx0], ist.inv_mass[idx1]
        gradient = ist.gradC[i, 0]
        if invM0 != 0.0:
            ist.dpos_withg[idx0] += invM0 * ist.lagrangian[i] * gradient 
        if invM1 != 0.0:
            ist.dpos_withg[idx1] -= invM1 * ist.lagrangian[i] * gradient

    for i in range(ist.NV):
        if ist.inv_mass[i] != 0.0:
            ist.dpos_withg[i] += ist.predict_pos[i] - ist.old_pos[i]


def transfer_back_to_pos_mfree(x):
    ist.dLambda.from_numpy(x)
    reset_dpos(ist.dpos)
    transfer_back_to_pos_mfree_kernel()
    update_pos(ist.inv_mass, ist.dpos, ist.pos, args.omega)

def spy_A(A,b):
    print("A:", A.shape, " b:", b.shape)
    scipy.io.mmwrite("A.mtx", A)
    plt.spy(A, markersize=1)
    plt.show()
    exit()





def is_symmetric(A):
    AT = A.transpose()
    diff = A - AT
    if diff.nnz == 0:
        return True
    maxdiff = np.max(np.abs(diff.data))
    return maxdiff < 1e-6

def csr_is_equal(A, B):
    if A.shape != B.shape:
        print("shape not equal")
        assert False
    diff = A - B
    if diff.nnz == 0:
        print("csr is equal! nnz=0")
        return True
    maxdiff = np.abs(diff.data).max()
    print("maxdiff: ", maxdiff)
    if maxdiff > 1e-6:
        assert False
    print("csr is equal!")
    return True

def dense_mat_is_equal(A, B):
    diff = A - B
    maxdiff = np.abs(diff).max()
    print("maxdiff: ", maxdiff)
    if maxdiff > 1e-6:
        assert False
    print("is equal!")
    return True





def fastFill_fetch():
    extlib.fastFillCloth_fetch_A_data(ist.spmat_data)
    A = scipy.sparse.csr_matrix((ist.spmat_data, ist.spmat_indices, ist.spmat_indptr), shape=(ist.NE, ist.NE))
    return A


def fastFillCloth_run():
    extlib.fastFillCloth_run(ist.pos.to_numpy())


@ti.kernel
def fill_A_diag_kernel(diags:ti.types.ndarray(dtype=ti.f32), alpha:ti.f32, inv_mass:ti.template(), edge:ti.template()):
    for i in range(edge.shape[0]):
        diags[i] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha


@ti.kernel
def fill_A_ijv_kernel(ii:ti.types.ndarray(dtype=ti.i32), jj:ti.types.ndarray(dtype=ti.i32), vv:ti.types.ndarray(dtype=ti.f32), num_adjacent_edge:ti.types.ndarray(dtype=ti.i32), adjacent_edge:ti.types.ndarray(dtype=ti.i32), adjacent_edge_abc:ti.types.ndarray(dtype=ti.i32),  inv_mass:ti.template(), alpha:ti.f32):
    n = 0
    ist.NE = ist.adjacent_edge.shape[0]
    ti.loop_config(serialize=True)
    for i in range(ist.NE): #对每个edge，找到所有的adjacent edge，填充到offdiag，然后填充diag
        for k in range(num_adjacent_edge[i]):
            ia = adjacent_edge[i,k]
            a = adjacent_edge_abc[i, k * 3]
            b = adjacent_edge_abc[i, k * 3 + 1]
            c = adjacent_edge_abc[i, k * 3 + 2]
            g_ab = (ist.pos[a] - ist.pos[b]).normalized()
            g_ac = (ist.pos[a] - ist.pos[c]).normalized()
            offdiag = inv_mass[a] * g_ab.dot(g_ac)
            if offdiag == 0:
                continue
            ii[n] = i
            jj[n] = ia
            vv[n] = offdiag
            n += 1
        # diag
        ii[n] = i
        jj[n] = i
        vv[n] = inv_mass[ist.edge[i][0]] + inv_mass[ist.edge[i][1]] + alpha
        n += 1 



@ti.kernel
def compute_potential_energy():
    ist.potential_energy[None] = 0.0
    ist.inv_alpha = 1.0/ist.compliance
    for i in range(ist.NE):
        ist.potential_energy[None] += 0.5 * ist.inv_alpha * ist.constraints[i]**2

@ti.kernel
def compute_inertial_energy():
    ist.inertial_energy[None] = 0.0
    ist.inv_h2 = 1.0 / ist.delta_t**2
    for i in range(ist.NV):
        if ist.inv_mass[i] == 0.0:
            continue
        ist.inertial_energy[None] += 0.5 / ist.inv_mass[i] * (ist.pos[i] - ist.predict_pos[i]).norm_sqr() * ist.inv_h2




def calc_conv(r):
    return (r[-1]/r[0])**(1.0/(len(r)-1))


def report_multilevel_details(Ps, num_levels):
    logging.info(f"    num_levels:{num_levels}")
    num_points_level = []
    for i in range(len(Ps)):
        num_points_level.append(Ps[i].shape[0])
    num_points_level.append(Ps[-1].shape[1])
    for i in range(num_levels):
        logging.info(f"    num points of level {i}: {num_points_level[i]}")


def should_setup():
    return ((ist.frame%args.setup_interval==0 or (args.restart==True and ist.frame==args.restart_frame)) and (ist.ite==0))



def AMG_calc_r(r, r0, tic_iter, r_Axb):
    tic = time.perf_counter()

    t_iter = perf_counter()-tic_iter
    tic_calcr = perf_counter()
    calc_dual_residual(ist.dual_residual, ist.edge, ist.rest_len, ist.lagrangian, ist.pos)
    dual_r = np.linalg.norm(ist.dual_residual.to_numpy()).astype(float)
    # compute_potential_energy()
    # compute_inertial_energy()
    # robj = (potential_energy[None]+inertial_energy[None])
    r_Axb = r_Axb.tolist()
    if args.use_PXPBD_v1 or args.use_PXPBD_v2:
        G = fill_G()
        primary_residual = calc_primary_residual(G, ist.M_inv)
        primal_r = np.linalg.norm(primary_residual).astype(float)
        Newton_r = np.linalg.norm(np.concatenate((ist.dual_residual.to_numpy(), primary_residual))).astype(float)

    # if export_fullr:
    #     fulldual_final = dual_residual.to_numpy()
    #     np.savez_compressed(args.out_dir+'/r/'+ f'fulldual_{ist.frame}-{ist.ite}', fulldual0, fulldual_final)

    logging.info(f"    convergence factor: {calc_conv(r_Axb):.2f}")
    logging.info(f"    Calc r time: {(perf_counter()-tic_calcr)*1000:.0f}ms")

    if args.export_log:
        logging.info(f"    iter total time: {t_iter*1000:.0f}ms")
        if args.use_PXPBD_v1 or args.use_PXPBD_v2:
            r.append(ResidualDataPrimal(dual_r, primal_r, Newton_r, len(r_Axb), t_iter))
            logging.info(f"{ist.frame}-{ist.ite} Newton:{Newton_r:.2e} rsys:{r_Axb[0]:.2e} {r_Axb[-1]:.2e} dual:{dual_r:.2e} primal:{primal_r:.2e} iter:{len(r_Axb)}")
        else:
            r.append(ResidualData(dual_r, len(r_Axb), t_iter))
            logging.info(f"{ist.frame}-{ist.ite} rsys:{r_Axb[0]:.2e} {r_Axb[-1]:.2e} dual0:{r0.dual:.2e} dual:{dual_r:.2e}  iter:{len(r_Axb)}")

    ist.t_export += perf_counter()-tic



def AMG_calc_r0():
    calc_dual_residual(ist.dual_residual, ist.edge, ist.rest_len, ist.lagrangian, ist.pos)
    dual_r = np.linalg.norm(ist.dual_residual.to_numpy()).astype(float)
    # compute_potential_energy()
    # compute_inertial_energy()
    # robj = (potential_energy[None]+inertial_energy[None])
    if args.use_PXPBD_v1 or args.use_PXPBD_v2:
        G = fill_G()
        primary_residual = calc_primary_residual(G, ist.M_inv)
        primal_r = np.linalg.norm(primary_residual).astype(float)
        Newton_r = np.linalg.norm(np.concatenate((ist.dual_residual.to_numpy(), primary_residual))).astype(float)

        r0 = (Residual0DataPrimal(dual_r, primal_r, Newton_r))
    else:
        r0 = (Residual0Data(dual_r))
    return r0


def do_export_r(r):
    tic = time.perf_counter()
    serialized_r = [r[i]._asdict() for i in range(len(r))]
    r_json = json.dumps(serialized_r)
    with open(args.out_dir+'/r/'+ f'{ist.frame}.json', 'w') as file:
        file.write(r_json)
    ist.t_export += time.perf_counter()-tic


@ti.kernel
def PXPBD_b_kernel(pos:ti.template(), predict_pos:ti.template(), lagrangian:ti.template(), inv_mass:ti.template(), gradC:ti.template(), b:ti.types.ndarray(), Minv_gg:ti.template()):
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]

        if invM0 != 0.0:
            Minv_gg[idx0] = invM0 * lagrangian[i] * gradC[i, 0] + (pos[idx0] - predict_pos[idx0])
        if invM1 != 0.0:
            Minv_gg[idx1] = invM1 * lagrangian[i] * gradC[i, 1] + (pos[idx0] - predict_pos[idx0])

    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        if invM1 != 0.0 and invM0 != 0.0:
            b[idx0] += gradC[i, 0] @ Minv_gg[idx0] + gradC[i, 1] @ Minv_gg[idx1]

        #     Minv_gg =  (pos.to_numpy().flatten() - predict_pos.to_numpy().flatten()) - M_inv @ G.transpose() @ lagrangian.to_numpy()
        #     b += G @ Minv_gg


# v1-mfree
def PXPBD_v1_mfree_transfer_back_to_pos(x, Minv_gg):
    ist.dLambda.from_numpy(x)
    reset_dpos(ist.dpos)
    PXPBD_v1_mfree_transfer_back_to_pos_kernel(Minv_gg)
    update_pos(ist.inv_mass, ist.dpos, ist.pos)


# v1-mfree
@ti.kernel
def PXPBD_v1_mfree_transfer_back_to_pos_kernel(Minv_gg:ti.template()):
    for i in range(ist.NE):
        idx0, idx1 = ist.edge[i]
        invM0, invM1 = ist.inv_mass[idx0], ist.inv_mass[idx1]

        delta_lagrangian = ist.dLambda[i]
        ist.lagrangian[i] += delta_lagrangian

        gradient = ist.gradC[i, 0]
        
        if invM0 != 0.0:
            ist.dpos[idx0] += invM0 * delta_lagrangian * gradient - Minv_gg[idx0]
        if invM1 != 0.0:
            ist.dpos[idx1] -= invM1 * delta_lagrangian * gradient - Minv_gg[idx1]
        


def AMG_setup_phase():
    tic = time.perf_counter()
    # A = fill_A_csr_ti() taichi version
    A = fastFill_fetch()
    ist.Ps = build_Ps(A,args,ist,extlib)
    ist.num_levels = len(ist.Ps)+1
    logging.info(f"    build_Ps time:{time.perf_counter()-tic}")

    extlib.fastmg_setup_nl(ist.num_levels)

    tic = time.perf_counter()
    update_P(ist.Ps)
    logging.info(f"    update_P time: {time.perf_counter()-tic:.2f}s")

    tic = time.perf_counter()
    cuda_set_A0(A)
    extlib.fastmg_setup_smoothers(1) # 1 means chebyshev
    logging.info(f"    setup smoothers time:{perf_counter()-tic}")

    report_multilevel_details(ist.Ps, ist.num_levels)


def AMG_RAP():
    tic3 = time.perf_counter()
    for lv in range(ist.num_levels-1):
        extlib.fastmg_RAP(lv) 
    logging.info(f"    RAP time: {(time.perf_counter()-tic3)*1000:.0f}ms")


# original XPBD dlam2dpos
def AMG_dlam2dpos(x):
    tic = time.perf_counter()
    transfer_back_to_pos_mfree(x)
    logging.info(f"    dlam2dpos time: {(perf_counter()-tic)*1000:.0f}ms")


# v1: with g, modify b and dpos
def AMG_PXPBD_v1_dlam2dpos(x,G, Minv_gg):
    dLambda_ = x.copy()
    ist.lagrangian.from_numpy(ist.lagrangian.to_numpy() + dLambda_)
    dpos = ist.M_inv @ G.transpose() @ dLambda_ 
    dpos -=  Minv_gg
    dpos = dpos.reshape(-1, 3)
    ist.pos.from_numpy(ist.pos_mid.to_numpy() + ist.omega*dpos)


# v2: blended, only modify dpos
def AMG_PXPBD_v2_dlam2dpos(x):
    tic = time.perf_counter()
    ist.dLambda.from_numpy(x)
    reset_dpos(ist.dpos)
    ist.dpos_withg.fill(0)
    transfer_back_to_pos_mfree_kernel()
    update_pos(ist.inv_mass, ist.dpos, ist.pos)
    compute_C_and_gradC_kernel(ist.pos, ist.gradC, ist.edge, ist.constraints, ist.rest_len) # required by dlam2dpos
    # G = fill_G()
    transfer_back_to_pos_mfree_kernel_withg()
    # dpos_withg_np = (predict_pos.to_numpy() - pos.to_numpy()).flatten() + M_inv @ G.transpose() @ lagrangian.to_numpy()
    # dpos_withg.from_numpy(dpos_withg_np.reshape(-1, 3))
    update_pos_blend(ist.inv_mass, ist.dpos, ist.pos, ist.dpos_withg)
    update_pos(ist.inv_mass, ist.dpos_withg, ist.pos)
    logging.info(f"    dlam2dpos time: {(perf_counter()-tic)*1000:.0f}ms")


def AMG_b():
    update_constraints_kernel(ist.pos, ist.edge, ist.rest_len, ist.constraints)
    b = -ist.constraints.to_numpy() - ist.alpha_tilde_np * ist.lagrangian.to_numpy()
    return b


def AMG_PXPBD_v1_b(G):
    # #we calc inverse mass times gg(primary residual), because NCONS may contains infinity for fixed pin points. And gg always appears with inv_mass.
    update_constraints_kernel(ist.pos, ist.edge, ist.rest_len, ist.constraints)
    b = -ist.constraints.to_numpy() - ist.alpha_tilde_np * ist.lagrangian.to_numpy()

    # PXPBD_b_kernel(pos, predict_pos, lagrangian, inv_mass, gradC, b, Minv_gg)
    MASS = sp.diags(1.0/(ist.M_inv.diagonal()+1e-12), format="csr")
    Minv_gg =  MASS@ist.M_inv@(ist.pos.to_numpy().flatten() - ist.predict_pos.to_numpy().flatten()) - ist.M_inv @ G.transpose() @ ist.lagrangian.to_numpy()
    b += G @ Minv_gg
    return b, Minv_gg
    

def AMG_A():
    tic2 = perf_counter()
    fastFillCloth_run()
    logging.info(f"    fill_A time: {(perf_counter()-tic2)*1000:.0f}ms")

def calc_dual():
    calc_dual_residual(ist.dual_residual, ist.edge, ist.rest_len, ist.lagrangian, ist.pos)
    return ist.dual_residual.to_numpy()


def substep_all_solver():
    tic1 = time.perf_counter()
    semi_euler(ist.old_pos, ist.inv_mass, ist.vel, ist.pos, ist.predict_pos, args.delta_t)
    reset_lagrangian(ist.lagrangian)
    r = [] # residual list of one ist.frame
    # fulldual0 = calc_dual()
    # print("dual0: ", np.linalg.norm(fulldual0))
    logging.info(f"pre-loop time: {(perf_counter()-tic1)*1000:.0f}ms")
    r0 = AMG_calc_r0()
    for ist.ite in range(args.maxiter):
        tic_iter = perf_counter()
        if args.use_PXPBD_v1:
            copy_field(ist.pos_mid, ist.pos)
        compute_C_and_gradC_kernel(ist.pos, ist.gradC, ist.edge, ist.constraints, ist.rest_len) # required by dlam2dpos
        b = AMG_b()
        if args.use_PXPBD_v1:
            G = fill_G()
            b, Minv_gg = AMG_PXPBD_v1_b(G)
        if not args.use_cuda:
            x, r_Axb = AMG_python(b,args,ist,fill_A_csr_ti,should_setup,copy_A=False)
        else:
            AMG_A()
            if args.export_matrix:
                A = fastFill_fetch()
                export_A_b(A, b, dir=args.out_dir + "/A/", postfix=f"F{ist.frame}",binary=args.export_matrix_binary)
            if should_setup():
                AMG_setup_phase()
            extlib.fastmg_set_A0_from_fastFillCloth()
            AMG_RAP()
            x, r_Axb = AMG_solve(b, maxiter=args.maxiter_Axb, tol=1e-5)
        if args.use_PXPBD_v1:
            AMG_PXPBD_v1_dlam2dpos(x, G, Minv_gg)
        elif args.use_PXPBD_v2:
            AMG_PXPBD_v2_dlam2dpos(x)
        else:
            AMG_dlam2dpos(x)
        AMG_calc_r(r, r0, tic_iter, r_Axb)
        logging.info(f"iter time(with export): {(perf_counter()-tic_iter)*1000:.0f}ms")

        if args.use_PXPBD_v1:
            rtol = 1e-1
            if  r[-1].Newton<rtol*r[0].Newton:
                break
        else:
            if r[-1].dual<args.tol:
                break

        if is_stall(r):
            logging.info("Stall detected, break")
            break
    
    tic = time.perf_counter()
    logging.info(f"n_outer: {ist.ite+1}")
    ist.n_outer_all.append(ist.ite+1)
    if args.export_residual:
        do_export_r(r)
    update_vel(ist.old_pos, ist.inv_mass, ist.vel, ist.pos)
    logging.info(f"post-loop time: {(time.perf_counter()-tic)*1000:.0f}ms")



@ti.kernel
def copy_field(dst: ti.template(), src: ti.template()):
    for i in src:
        dst[i] = src[i]




@ti.kernel
def init_scale():
    scale = 1.5
    for i in range(ist.NV):
        ist.pos[i] *= scale



def print_all_globals(global_vars):
    logging.info("\n\n### Global Variables ###")
    import sys
    module_name = sys.modules[__name__].__name__
    global_vars = global_vars.copy()
    keys_to_delete = []
    for var_name, var_value in global_vars.items():
        if var_name != module_name and not var_name.startswith('__') and not callable(var_value) and not isinstance(var_value, type(sys)):
            if var_name == 'parser':
                continue
            if args.export_log:
                logging.info(f"{var_name} = {var_value}")
            keys_to_delete.append(var_name)
    logging.info("\n\n\n")




def dict_to_ndarr(d:dict)->np.ndarray:
    lengths = np.array([len(v) for v in d.values()])

    max_len = max(len(item) for item in d.values())
    # 使用填充或截断的方式转换为NumPy数组
    arr = np.array([list(item) + [-1]*(max_len - len(item)) if len(item) < max_len else list(item)[:max_len] for item in d.values()])
    return arr, lengths



# ---------------------------------------------------------------------------- #
#                                 start fill A                                 #
# ---------------------------------------------------------------------------- #
class FillACloth():
    def load(self):
        self.cache_and_initFill()

    def initFill_python(self):
        tic1 = perf_counter()
        print("Initializing adjacent edge and abc...")
        ist.adjacent_edge, v2e_dict = self.init_adj_edge(edges=ist.edge.to_numpy())
        ist.adjacent_edge,ist.num_adjacent_edge = dict_to_ndarr(ist.adjacent_edge)
        v2e_np, num_v2e = dict_to_ndarr(v2e_dict)

        ist.adjacent_edge_abc = np.empty((ist.NE, 20*3), dtype=np.int32)
        ist.adjacent_edge_abc.fill(-1)
        self.init_adjacent_edge_abc_kernel(ist.NE,ist.edge,ist.adjacent_edge,ist.num_adjacent_edge,ist.adjacent_edge_abc)

        ist.num_nonz = self.calc_num_nonz(ist.num_adjacent_edge) 
        data, indices, indptr = self.init_A_CSR_pattern(ist.num_adjacent_edge, ist.adjacent_edge)
        ii, jj = self.csr_index_to_coo_index(indptr, indices)
        print(f"initFill time: {perf_counter()-tic1:.3f}s")
        return ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, data, indices, indptr, ii, jj, v2e_np, num_v2e

    def initFill_cpp(self):
        tic1 = perf_counter()
        print("Initializing adjacent edge and abc...")
        extlib.initFillCloth_set(ist.edge.to_numpy(), ist.NE)
        extlib.initFillCloth_run()
        ist.num_nonz = extlib.initFillCloth_get_nnz()

        MAX_ADJ = 20
        MAX_V2E = MAX_ADJ
        ist.adjacent_edge = np.zeros((ist.NE, MAX_ADJ), dtype=np.int32)
        ist.num_adjacent_edge = np.zeros(ist.NE, dtype=np.int32)
        ist.adjacent_edge_abc = np.zeros((ist.NE, MAX_ADJ*3), dtype=np.int32)
        ist.spmat_data = np.zeros(ist.num_nonz, dtype=np.float32)
        ist.spmat_indices = np.zeros(ist.num_nonz, dtype=np.int32)
        ist.spmat_indptr = np.zeros(ist.NE+1, dtype=np.int32)
        ist.spmat_ii = np.zeros(ist.num_nonz, dtype=np.int32)
        ist.spmat_jj = np.zeros(ist.num_nonz, dtype=np.int32)
        ist.v2e = np.zeros((ist.NV, MAX_V2E), dtype=np.int32)
        ist.num_v2e = np.zeros(ist.NV, dtype=np.int32)

        extlib.initFillCloth_get(ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, ist.spmat_indices, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj, ist.v2e, ist.num_v2e)
        print(f"initFill time: {perf_counter()-tic1:.3f}s")
        return ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, ist.spmat_data, ist.spmat_indices, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj, ist.v2e, ist.num_v2e

    def cache_and_initFill(self):
        if  os.path.exists(f'cache_initFill_N{N}.npz') and args.use_cache:
            npzfile= np.load(f'cache_initFill_N{N}.npz')
            (ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, ist.spmat_data, ist.spmat_indices, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj) = (npzfile[key] for key in ['adjacent_edge', 'num_adjacent_edge', 'adjacent_edge_abc', 'num_nonz', 'spmat_data', 'spmat_indices', 'spmat_indptr', 'spmat_ii', 'spmat_jj'])
            ist.num_nonz = int(ist.num_nonz) # npz save int as np.array, it will cause bug in taichi kernel
            print(f"load cache_initFill_N{N}.npz")
        else:
            if args.use_cuda and args.use_cpp_initFill:
                initFill = self.initFill_cpp
            else:
                initFill = self.initFill_python
            ist.adjacent_edge, ist.num_adjacent_edge, ist.adjacent_edge_abc, ist.num_nonz, ist.spmat_data, ist.spmat_indices, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj, ist.v2e, ist.num_v2e = initFill()
            print("caching init fill...")
            tic = perf_counter() # savez_compressed will save 10x space(1.4G->140MB), but much slower(33s)
            np.savez(f'cache_initFill_N{N}.npz', adjacent_edge=ist.adjacent_edge, num_adjacent_edge=ist.num_adjacent_edge, adjacent_edge_abc=ist.adjacent_edge_abc, num_nonz=ist.num_nonz, spmat_data=ist.spmat_data, spmat_indices=ist.spmat_indices, spmat_indptr=ist.spmat_indptr, spmat_ii=ist.spmat_ii, spmat_jj=ist.spmat_jj)
            print("time of caching:", perf_counter()-tic)

    def calc_num_nonz(num_adjacent_edge):
        ist.num_nonz = np.sum(ist.num_adjacent_edge)+num_adjacent_edge.shape[0]
        return ist.num_nonz

    def calc_nnz_each_row(num_adjacent_edge):
        nnz_each_row = ist.num_adjacent_edge[:] + 1
        return nnz_each_row

    def init_A_CSR_pattern(num_adjacent_edge, adjacent_edge):
        num_adj = ist.num_adjacent_edge
        adj = ist.adjacent_edge
        nonz = np.sum(num_adj)+ist.NE
        indptr = np.zeros(ist.NE+1, dtype=np.int32)
        indices = np.zeros(nonz, dtype=np.int32)
        data = np.zeros(nonz, dtype=np.float32)

        indptr[0] = 0
        for i in range(0,ist.NE):
            num_adj_i = num_adj[i]
            indptr[i+1]=indptr[i] + num_adj_i + 1
            indices[indptr[i]:indptr[i+1]-1]= adj[i][:num_adj_i]
            indices[indptr[i+1]-1]=i

        assert indptr[-1] == nonz

        return data, indices, indptr


    def csr_index_to_coo_index(indptr, indices):
        ii, jj = np.zeros_like(indices), np.zeros_like(indices)
        for i in range(ist.NE):
            ii[indptr[i]:indptr[i+1]]=i
            jj[indptr[i]:indptr[i+1]]=indices[indptr[i]:indptr[i+1]]
        return ii, jj


    

    
    def load_cache_initFill_to_cuda(self):
        self.cache_and_initFill()
        extlib.fastFillCloth_set_data(ist.edge.to_numpy(), ist.NE, ist.inv_mass.to_numpy(), ist.NV, ist.pos.to_numpy(), ist.alpha)
        extlib.fastFillCloth_init_from_python_cache(ist.adjacent_edge,
                                            ist.num_adjacent_edge,
                                            ist.adjacent_edge_abc,
                                            ist.num_nonz,
                                            ist.spmat_data,
                                            ist.spmat_indices,
                                            ist.spmat_indptr,
                                            ist.spmat_ii,
                                            ist.spmat_jj,
                                            ist.NE,
                                           ist.NV)




    def compare_find_shared_v_order(v,e1,e2,edge):
        # which is shared v in e1? 0 or 1
        order_in_e1 = 0 if edge[e1][0] == v else 1
        order_in_e2 = 0 if edge[e2][0] == v else 1
        return order_in_e1, order_in_e2


    # legacy
    def fill_A_by_spmm(M_inv, ALPHA):
        tic = time.perf_counter()
        G_ii, G_jj, G_vv = np.zeros(ist.NCONS*6, dtype=np.int32), np.zeros(ist.NCONS*6, dtype=np.int32), np.zeros(ist.NCONS*6, dtype=np.float32)
        fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, ist.gradC, ist.edge)
        G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(ist.NCONS, 3 * ist.NV))
        print(f"fill_G: {time.perf_counter() - tic:.4f}s")

        tic = time.perf_counter()
        if args.use_geometric_stiffness:
            # Geometric Stiffness: K = ist.NCONS - H, we only use diagonal of H and then replace M_inv with K_inv
            # https://github.com/FantasyVR/magicMirror/blob/a1e56f79504afab8003c6dbccb7cd3c024062dd9/geometric_stiffness/meshComparison/meshgs_SchurComplement.py#L143
            # https://team.inria.fr/imagine/files/2015/05/final.pdf eq.21
            # https://blog.csdn.net/weixin_43940314/article/details/139448858
            ist.K_diag.fill(0.0)
            compute_K_kernel(ist.K_diag)
            mass = 1.0/(M_inv.diagonal()+1e-12)
            MK_inv = scipy.sparse.diags([1.0/(mass - ist.K_diag)], [0], format="dia")
            M_inv = MK_inv # replace old M_inv with MK_inv

        A = G @ M_inv @ G.transpose() + ALPHA
        A = scipy.sparse.csr_matrix(A)
        # print("fill_A_by_spmm  time: ", time.perf_counter() - tic)
        return A, G
    
    
    @ti.kernel
    def init_adjacent_edge_abc_kernel(NE:int, edge:ti.template(), adjacent_edge:ti.types.ndarray(), num_adjacent_edge:ti.types.ndarray(), adjacent_edge_abc:ti.types.ndarray()):
        for i in range(NE):
            ii0 = edge[i][0]
            ii1 = edge[i][1]

            num_adj = num_adjacent_edge[i]
            for j in range(num_adj):
                ia = adjacent_edge[i,j]
                if ia == i:
                    continue
                jj0,jj1 = edge[ia]
                a, b, c = -1, -1, -1
                if ii0 == jj0:
                    a, b, c = ii0, ii1, jj1
                elif ii0 == jj1:
                    a, b, c = ii0, ii1, jj0
                elif ii1 == jj0:
                    a, b, c = ii1, ii0, jj1
                elif ii1 == jj1:
                    a, b, c = ii1, ii0, jj0
                adjacent_edge_abc[i, j*3] = a
                adjacent_edge_abc[i, j*3+1] = b
                adjacent_edge_abc[i, j*3+2] = c



    def init_adj_edge(edges: np.ndarray):
        # 构建数据结构
        vertex_to_edges = {}
        for edge_index, (v1, v2) in enumerate(edges):
            if v1 not in vertex_to_edges:
                vertex_to_edges[v1] = set()
            if v2 not in vertex_to_edges:
                vertex_to_edges[v2] = set()
            
            vertex_to_edges[v1].add(edge_index)
            vertex_to_edges[v2].add(edge_index)

        # 初始化存储所有边的邻接边的字典
        all_adjacent_edges = {}

        # 查找并存储每条边的邻接边
        for edge_index in range(len(edges)):
            v1, v2 = edges[edge_index]
            adjacent_edges = vertex_to_edges[v1] | vertex_to_edges[v2]  # 合并两个集合
            adjacent_edges.remove(edge_index)  # 移除边本身
            all_adjacent_edges[edge_index] = list(adjacent_edges)

        return all_adjacent_edges, vertex_to_edges

    # # 示例用法
    # edges = np.array([[0, 1], [1, 2], [2, 0], [1, 3]])
    # adjacent_edges_dict = init_adj_edge(edges)
    # print(adjacent_edges_dict)

# for cnt version, require init_A_CSR_pattern() to be called first
@ti.kernel
def fill_A_CSR_kernel(data:ti.types.ndarray(dtype=ti.f32), 
                    indptr:ti.types.ndarray(dtype=ti.i32), 
                    ii:ti.types.ndarray(dtype=ti.i32), 
                    jj:ti.types.ndarray(dtype=ti.i32),
                    adjacent_edge_abc:ti.types.ndarray(dtype=ti.i32),
                    num_nonz:ti.i32,
                    alpha:ti.f32):
    for cnt in range(num_nonz):
        i = ii[cnt] # row index
        j = jj[cnt] # col index
        k = cnt - indptr[i] #k-th non-zero element of i-th row. 
        # Because the diag is the final element of each row, 
        # it is also the k-th adjacent edge of i-th edge.
        if i == j: # diag
            data[cnt] = ist.inv_mass[ist.edge[i][0]] + ist.inv_mass[ist.edge[i][1]] + alpha
            continue
        a = adjacent_edge_abc[i, k * 3]
        b = adjacent_edge_abc[i, k * 3 + 1]
        c = adjacent_edge_abc[i, k * 3 + 2]
        g_ab = (ist.pos[a] - ist.pos[b]).normalized()
        g_ac = (ist.pos[a] - ist.pos[c]).normalized()
        offdiag = ist.inv_mass[a] * g_ab.dot(g_ac)
        data[cnt] = offdiag

def fill_A_csr_ti(ist):
    fill_A_CSR_kernel(ist.spmat_data, ist.spmat_indptr, ist.spmat_ii, ist.spmat_jj, ist.adjacent_edge_abc, ist.num_nonz, ist.alpha)
    A = scipy.sparse.csr_matrix((ist.spmat_data, ist.spmat_indices, ist.spmat_indptr), shape=(ist.NE, ist.NE))
    return A

def fill_G():
    tic = time.perf_counter()
    compute_C_and_gradC_kernel(ist.pos, ist.gradC, ist.edge, ist.constraints, ist.rest_len)
    G_ii, G_jj, G_vv = np.zeros(ist.NCONS*6, dtype=np.int32), np.zeros(ist.NCONS*6, dtype=np.int32), np.zeros(ist.NCONS*6, dtype=np.float32)
    fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, ist.gradC, ist.edge)
    G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(ist.NCONS, 3 * ist.NV))
    print(f"    fill_G: {time.perf_counter() - tic:.4f}s")
    return G

# ---------------------------------------------------------------------------- #
#                                  end fill A                                  #
# ---------------------------------------------------------------------------- #

def ending(timer_loop, start_date, initial_frame):
    t_all = time.perf_counter() - timer_loop
    end_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.end_frame = ist.frame

    len_n_outer_all = len(ist.n_outer_all) if len(ist.n_outer_all) > 0 else 1
    sum_n_outer = sum(ist.n_outer_all)
    avg_n_outer = sum_n_outer / len_n_outer_all
    max_n_outer = max(ist.n_outer_all)
    max_n_outer_index = ist.n_outer_all.index(max_n_outer)

    n_outer_all_np = np.array(ist.n_outer_all, np.int32)    
    np.savetxt(args.out_dir+"/r/n_outer.txt", n_outer_all_np, fmt="%d")

    sim_time_with_export = time.perf_counter() - timer_loop
    sim_time = sim_time_with_export - ist.t_export_total
    avg_sim_time = sim_time / (args.end_frame - initial_frame)


    s = f"\n-------\n"+\
    f"Time: {(sim_time):.2f}s = {(sim_time)/60:.2f}min.\n" + \
    f"Time with exporting: {(sim_time_with_export):.2f}s = {sim_time_with_export/60:.2f}min.\n" + \
    f"Frame {initial_frame}-{args.end_frame}({args.end_frame-initial_frame} frames)."+\
    f"\nAvg: {avg_sim_time}s/ist.frame."+\
    f"\nStart\t{start_date},\nEnd\t{end_date}."+\
    f"\nTime of exporting: {ist.t_export_total:.3f}s" + \
    f"\nSum n_outer: {sum_n_outer} \nAvg n_outer: {avg_n_outer:.1f}"+\
    f"\nMax n_outer: {max_n_outer} \nMax n_outer ist.frame: {max_n_outer_index + initial_frame}." + \
    f"\nstalled at {all_stalled}"+\
    f"\nCloth-N{N}" + \
    f"\ndt={args.delta_t}" + \
    f"\nSolver: {args.solver_type}" + \
    f"\nout_dir: {args.out_dir}" 

    logging.info(s)

    start_date = start_date.strftime("%Y-%m-%d-%H-%M-%S")
    out_dir_name = Path(args.out_dir).name
    name = start_date + "_" +  str(out_dir_name) 
    file_name = f"result/meta/{name}.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(s)

    file_name2 = f"{args.out_dir}/meta.txt"
    with open(file_name2, "w", encoding="utf-8") as file:
        file.write(s)




# ---------------------------------------------------------------------------- #
#                                initialization                                #
# ---------------------------------------------------------------------------- #
def init():
    process_dirs(args)
    
    log_level = logging.INFO
    if not args.export_log:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format="%(message)s",filename=args.out_dir + f'/latest.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"args.out_dir: {args.out_dir}")


    # print_all_globals(global_vars)

    global ist
    ist = Cloth()

    tic_init = time.perf_counter()
    ist.start_wall_time = datetime.datetime.now()
    logging.info(f"start wall time:{ist.start_wall_time}")

    logging.info("\nInitializing...")
    logging.info("Initializing pos..")
    init_pos(ist.inv_mass,ist.pos, args.N, ist.NV)
    init_tri(ist.tri)
    init_edge(ist.edge, ist.rest_len, ist.pos)
    if args.use_bending:
        ist.tri_pairs, ist.bending_length = init_bending(ist.tri.to_numpy().reshape(-1, 3), ist.pos)
    if args.setup_num == 1:
        init_scale()
    write_mesh(args.out_dir + f"/mesh/{ist.frame:04d}", ist.pos.to_numpy(), ist.tri.to_numpy())
    logging.info("Initializing pos and edge done")

    tic = time.perf_counter()
    if args.solver_type == "AMG":
        ist.fill_A = FillACloth()
        if args.use_cuda:
            ist.fill_A.load_cache_initFill_to_cuda()
        else:
            ist.fill_A.cache_and_initFill()
    logging.info(f"Init fill time: {time.perf_counter()-tic:.3f}s")

    # # colors, num_colors = graph_coloring(edge.to_numpy())
    # logging.info(f"start coloring")
    # tic = time.perf_counter()
    # colors = greedy_coloring(edge.to_numpy())
    # logging.info(f"end coloring, time: {time.perf_counter()-tic:.3f}s")

    if args.restart:
        do_restart()

    inv_mass_np = np.repeat(ist.inv_mass.to_numpy(), 3, axis=0)
    ist.M_inv = scipy.sparse.diags(inv_mass_np)
    ist.alpha_tilde_np = np.array([ist.alpha] * ist.NCONS)
    ist.ALPHA = scipy.sparse.diags(ist.alpha_tilde_np)

    logging.info(f"Initialization done. Cost time:  {time.perf_counter() - tic_init:.3f}s") 


def export_after_substep():
    ist.tic_export = time.perf_counter()
    if args.export_mesh:
        write_mesh(args.out_dir + f"/mesh/{ist.frame:04d}", ist.pos.to_numpy(), ist.tri.to_numpy())
    if args.export_state:
        save_state(args.out_dir+'/state/' + f"{ist.frame:04d}.npz")
    ist.t_export += time.perf_counter()-ist.tic_export
    ist.t_export_total += ist.t_export
    t_frame = time.perf_counter()-ist.tic_frame
    if args.export_log:
        logging.info(f"Time of exporting: {ist.t_export:.3f}s")
        logging.info(f"Time of ist.frame-{ist.frame}: {t_frame:.3f}s")

class Viewer:
    if args.use_viewer:
        window = ti.ui.Window("Display Mesh", (1024, 1024))
        canvas = window.get_canvas()
        canvas.set_background_color((1, 1, 1))
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        # camera.position(0.5, 0.4702609, 1.52483202)
        # camera.lookat(0.5, 0.9702609, -0.97516798)
        camera.position(0.5, 0.0, 2.5)
        camera.lookat(0.5, 0.5, 0.0)
        camera.fov(90)
        gui = window.get_gui()
    
    def do_render_taichi(viewer):
        if args.use_viewer:
            viewer.camera.track_user_inputs(viewer.window, movement_speed=0.003, hold_key=ti.ui.RMB)
            viewer.scene.set_camera(viewer.camera)
            viewer.scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
            viewer.scene.mesh(ist.pos, ist.tri, color=(1.0,0,0), two_sided=True)
            viewer.canvas.scene(viewer.scene)
            # you must call this function, even if we just want to save the image, otherwise the GUI image will not update.
            viewer.window.show()
            if args.save_image:
                file_path = args.out_dir + f"{ist.frame:04d}.png"
                viewer.window.save_image(file_path)  # export and show in GUI
        
    def do_render_control(viewer):
        if args.use_viewer:
            for e in viewer.window.get_events(ti.ui.PRESS):
                if e.key in [ti.ui.ESCAPE]:
                    exit()
                if e.key == ti.ui.SPACE:
                    paused = not paused
                    print("paused:",paused)

viewer = Viewer()


def run():
    timer_loop = time.perf_counter()
    initial_frame = ist.frame
    step_pbar = tqdm.tqdm(total=args.end_frame, initial=ist.frame)
    ist.t_export_total = 0.0

    try:
        while True:
            ist.tic_frame = time.perf_counter()
            ist.t_export = 0.0

            viewer.do_render_control()

            if not ist.paused:
                if args.solver_type == "XPBD":
                    substep_xpbd()
                else:
                    substep_all_solver()
                ist.frame += 1
                
                export_after_substep()

            if ist.frame == args.end_frame:
                print("Normallly end.")
                ending(timer_loop, ist.start_wall_time, initial_frame)
                exit()

            if args.use_viewer:
                viewer.do_render_taichi(viewer)

            logging.info("\n")
            step_pbar.update(1)
            logging.info("")

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        ending(timer_loop, ist.start_wall_time, initial_frame)


def main():
    init()
    run()

if __name__ == "__main__":
    main()
