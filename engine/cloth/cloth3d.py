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
from pyamg.relaxation.smoothing import approximate_spectral_radius, chebyshev_polynomial_coefficients
from time import perf_counter
import pyamg
import numpy.ctypeslib as ctl


prj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"

# parameters not in argparse
frame = 0
ite=0
save_image = True
paused = False
save_P, load_P = False, False
use_viewer = False
export_mesh = True
use_PXPBD_v1 = False
use_geometric_stiffness = False
export_fullr = False
calc_r_xpbd = True
num_levels = 0
t_export = 0.0
use_cpp_initFill = True
PXPBD_ksi = 1.0


#parse arguments to change default values
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1024)
parser.add_argument("-delta_t", type=float, default=1e-3)
parser.add_argument("-solver_type", type=str, default='AMG', help='"AMG", "GS", "XPBD"')
parser.add_argument("-export_matrix", type=int, default=False)
parser.add_argument("-export_matrix_binary", type=int, default=True)
parser.add_argument("-export_state", type=int, default=False)
parser.add_argument("-export_residual", type=int, default=True)
parser.add_argument("-end_frame", type=int, default=100)
parser.add_argument("-out_dir", type=str, default=f"result/latest/")
parser.add_argument("-auto_another_outdir", type=int, default=False)
parser.add_argument("-restart", type=int, default=False)
parser.add_argument("-restart_frame", type=int, default=21)
parser.add_argument("-restart_dir", type=str, default="result/meta/")
parser.add_argument("-restart_from_last_frame", type=int, default=False)
parser.add_argument("-maxiter", type=int, default=1000)
parser.add_argument("-maxiter_Axb", type=int, default=100)
parser.add_argument("-export_log", type=int, default=True)
parser.add_argument("-setup_num", type=int, default=0, help="attach:0, scale:1")
parser.add_argument("-use_json", type=int, default=False, help="json configs will overwrite the command line args")
parser.add_argument("-json_path", type=str, default="data/scene/cloth/config.json", help="json configs will overwrite the command line args")
parser.add_argument("-arch", type=str, default="cpu", help="taichi arch: gpu or cpu")
parser.add_argument("-use_cuda", type=int, default=True)
parser.add_argument("-cuda_dir", type=str, default="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")
parser.add_argument("-smoother_type", type=str, default="chebyshev")
parser.add_argument("-use_cache", type=int, default=True)
parser.add_argument("-setup_interval", type=int, default=20)


args = parser.parse_args()
N = args.N
delta_t = args.delta_t
gravity = [0.0, -9.8, 0.0]
if args.setup_num==1: gravity = [0.0, 0.0, 0.0]
else : gravity = [0.0, -9.8, 0.0]
out_dir =  args.out_dir + "/"


def parse_json_params(path, vars_to_overwrite):
    if not os.path.exists(path):
        assert False, f"json file {path} not exist!"
    print(f"CAUTION: using json config file {path} to overwrite the command line args!")
    with open(path, "r") as json_file:
        config = json.load(json_file)
    for key, value in config.items():
        if key in vars_to_overwrite:
            if vars_to_overwrite[key] != value:
                print(f"overwriting {key} from {vars_to_overwrite[key]} to {value}")
                vars_to_overwrite[key] = value
        else:
            print(f"json key {key} not exist in vars_to_overwrite!")

if args.use_json:
    parse_json_params(args.json_path, globals())


#to print out the parameters
global_vars = globals().copy()



if args.arch == "gpu":
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)


NV = (N + 1)**2
NT = 2 * N**2
NE = 2 * N * (N + 1) + N**2
NCONS = NE
new_M = int(NE / 100)
compliance = 1.0e-8  #see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
alpha = compliance * (1.0 / delta_t / delta_t)  # timestep related compliance, see XPBD paper
omega = 0.5

tri = ti.field(ti.i32, shape=3 * NT)
edge        = ti.Vector.field(2, dtype=int, shape=(NE))
pos         = ti.Vector.field(3, dtype=float, shape=(NV))
dpos        = ti.Vector.field(3, dtype=float, shape=(NV))
dpos_withg  = ti.Vector.field(3, dtype=float, shape=(NV))
old_pos     = ti.Vector.field(3, dtype=float, shape=(NV))
vel         = ti.Vector.field(3, dtype=float, shape=(NV))
pos_mid     = ti.Vector.field(3, dtype=float, shape=(NV))
inv_mass    = ti.field(dtype=float, shape=(NV))
rest_len    = ti.field(dtype=float, shape=(NE))
lagrangian  = ti.field(dtype=float, shape=(NE))  
constraints = ti.field(dtype=float, shape=(NE))  
dLambda     = ti.field(dtype=float, shape=(NE))
# numerator   = ti.field(dtype=float, shape=(NE))
# denominator = ti.field(dtype=float, shape=(NE))
gradC       = ti.Vector.field(3, dtype = ti.float32, shape=(NE,2)) 
edge_center = ti.Vector.field(3, dtype = ti.float32, shape=(NE))
dual_residual       = ti.field(shape=(NE),    dtype = ti.float32) # -C - alpha * lagrangian
nnz_each_row = np.zeros(NE, dtype=int)
potential_energy = ti.field(dtype=float, shape=())
inertial_energy = ti.field(dtype=float, shape=())
predict_pos = ti.Vector.field(3, dtype=float, shape=(NV))
# primary_residual = np.zeros(dtype=float, shape=(3*NV))
# K = ti.Matrix.field(3, 3, float, (NV, NV)) 
# geometric stiffness, only retain diagonal elements
K_diag = np.zeros((NV*3), dtype=float)
# Minv_gg = ti.Vector.field(3, dtype=float, shape=(NV))



if use_PXPBD_v1:
    ResidualDataPrimal = namedtuple('residual', ['dual','primal','Newton', 'ninner','t'])
    Residual0DataPrimal = namedtuple('residual', ['dual','primal','Newton'])
else:
    ResidualData = namedtuple('residual', ['dual', 'ninner','t'])
    Residual0Data = namedtuple('residual', ['dual'])

n_outer_all = []


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
    extlib.fastmg_setup_jacobi.argtypes = [ctypes.c_float, ctypes.c_size_t]
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


class SpMat():
    def __init__(self, nnz, nrow=NE):
        self.nrow = nrow #number of rows
        self.nnz = nnz # number of non-zeros

        # csr format and coo format storage data
        self.ii = np.zeros(nnz, dtype=np.int32) #coo.row
        self.jj = np.zeros(nnz, dtype=np.int32) #coo.col or csr.indices
        self.data = np.zeros(nnz, dtype=np.float32) #coo.data
        self.indptr = np.zeros(nrow+1, dtype=np.int32) # csr.indptr, start index of each row

        # number of non-zeros in each row
        self.nnz_row = np.zeros(nrow, dtype=np.int32) 

        self.diags = np.zeros(nrow, dtype=np.float32)

        # i: row index,
        # j: col index,
        # k: non-zero index in i-th row,
        # n: non-zero index in data

    def _init_pattern(self):
        ii, jj, indptr, nnz_row, data = self.ii, self.jj, self.indptr, self.nnz_row, self.data

        num_adj = num_adjacent_edge.to_numpy()
        adj = adjacent_edge.to_numpy()
        indptr[0] = 0
        for i in range(self.nrow):
            nnz_row[i] = num_adj[i] + 1
            indptr[i+1]= indptr[i] + nnz_row[i]
            jj[indptr[i]:indptr[i+1]-1]= adj[i][:nnz_row[i]-1] #offdiag
            jj[indptr[i+1]-1]=i #diag
            ii[indptr[i]:indptr[i+1]]=i

        # scipy coo to csr transferring may loose zeros,
        #  so we fill a small value to prevent it.
        data.fill(-1e-9) 

    def ik2n(self, i, k):
        n = self.indptr[i] + k
        return n
    
    def ik2j(self, i, k):
        j = self.jj[self.indptr[i] + k]
        return j
    
    # This is slow, because we have to search and compare.
    # -1 means not found(not in the matrix)
    def ij2n(self,i,j):
        for n in range(self.indptr[i], self.indptr[i+1]):
            if self.jj[n] == j:
                return n
        return -1


@ti.kernel
def init_pos(
    inv_mass:ti.template(),
    pos:ti.template(),
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
    for i in range(NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        rest_len[i] = (p1 - p2).norm()

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


def read_tri_cloth(filename):
    edge_file_name = filename + ".edge"
    node_file_name = filename + ".node"
    face_file_name = filename + ".face"

    with open(node_file_name, "r") as f:
        lines = f.readlines()
        NV = int(lines[0].split()[0])
        pos = np.zeros((NV, 3), dtype=np.float32)
        for i in range(NV):
            pos[i] = np.array(lines[i + 1].split()[1:], dtype=np.float32)

    with open(edge_file_name, "r") as f:
        lines = f.readlines()
        NE = int(lines[0].split()[0])
        edge_indices = np.zeros((NE, 2), dtype=np.int32)
        for i in range(NE):
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
):
    g = ti.Vector(gravity)
    for i in range(NV):
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
    for i in range(NE):
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
def solve_constraints_xpbd(
    dual_residual: ti.template(),
    inv_mass:ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
):
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        delta_lagrangian = -(constraint + lagrangian[i] * alpha) / (invM0 + invM1 + alpha)
        lagrangian[i] += delta_lagrangian

        # residual
        dual_residual[i] = -(constraint + alpha * lagrangian[i])
        
        if invM0 != 0.0:
            dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient

@ti.kernel
def update_pos(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
):
    for i in range(NV):
        if inv_mass[i] != 0.0:
            pos[i] += omega * dpos[i]


@ti.kernel
def update_pos_blend(
    inv_mass:ti.template(),
    dpos:ti.template(),
    pos:ti.template(),
    dpos_withg:ti.template(),
):
    for i in range(NV):
        if inv_mass[i] != 0.0:
            pos[i] += omega *((1-PXPBD_ksi) * dpos[i] + PXPBD_ksi * dpos_withg[i])


@ti.kernel
def update_vel(
    old_pos:ti.template(),
    inv_mass:ti.template(),    
    vel:ti.template(),
    pos:ti.template(),
):
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / delta_t


@ti.kernel 
def reset_dpos(dpos:ti.template()):
    for i in range(NV):
        dpos[i] = ti.Vector([0.0, 0.0, 0.0])



@ti.kernel
def calc_dual_residual(
    dual_residual: ti.template(),
    edge:ti.template(),
    rest_len:ti.template(),
    lagrangian:ti.template(),
    pos:ti.template(),
):
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]

        # residual(lagrangian=0 for first iteration)
        dual_residual[i] = -(constraint + alpha * lagrangian[i])

def calc_primary_residual(G,M_inv):
    MASS = sp.diags(1.0/(M_inv.diagonal()+1e-12), format="csr")
    primary_residual = MASS @ (predict_pos.to_numpy().flatten() - pos.to_numpy().flatten()) - G.transpose() @ lagrangian.to_numpy()
    where_zeros = np.where(M_inv.diagonal()==0)
    primary_residual = np.delete(primary_residual, where_zeros)
    return primary_residual


def xpbd_calcr(tic_iter, fulldual0, r):
    global ite, frame, t_export
    tic_calcr = perf_counter()
    t_iter = perf_counter()-tic_iter
    dualr = np.linalg.norm(dual_residual.to_numpy()).astype(float)
    
    if export_fullr:
        np.savez(out_dir+'/r/'+ f'fulldual_{frame}-{ite}', fulldual0)

    dualr0 = np.linalg.norm(fulldual0).astype(float)

    if export_fullr:
        np.savez(out_dir+'/r/'+ f'fulldual_{frame}-{ite}', fulldual0)

    r.append(ResidualData(dualr, 1, t_iter))
    if args.export_log:
        logging.info(f"{frame}-{ite}  dualr0:{dualr0:.2e} dual:{dualr:.2e}  t:{t_iter:.2e}s calcr:{perf_counter()-tic_calcr:.2e}s")
    t_export += perf_counter() - tic_calcr
    return dualr, dualr0



def substep_xpbd():
    global ite, t_export, n_outer_all
    semi_euler(old_pos, inv_mass, vel, pos)
    reset_lagrangian(lagrangian)

    calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
    fulldual0 = dual_residual.to_numpy()

    r = []
    for ite in range(args.maxiter):
        tic_iter = perf_counter()

        reset_dpos(dpos)
        solve_constraints_xpbd(dual_residual, inv_mass, edge, rest_len, lagrangian, dpos, pos)
        update_pos(inv_mass, dpos, pos)

        if calc_r_xpbd:
            dualr, dualr0 = xpbd_calcr(tic_iter, fulldual0, r)

        if dualr < 0.1*dualr0 or dualr<1e-5:
            break
    n_outer_all.append(ite+1)

    if args.export_residual:
        do_export_r(r)
    update_vel(old_pos, inv_mass, vel, pos)



# ---------------------------------------------------------------------------- #
#                                build hierarchy                               #
# ---------------------------------------------------------------------------- #
@ti.kernel
def compute_R_based_on_kmeans_label_triplets(
    labels: ti.types.ndarray(dtype=int),
    ii: ti.types.ndarray(dtype=int),
    jj: ti.types.ndarray(dtype=int),
    vv: ti.types.ndarray(dtype=int),
    new_M: ti.i32,
    NCONS: ti.i32
):
    cnt=0
    ti.loop_config(serialize=True)
    for i in range(new_M):
        for j in range(NCONS):
            if labels[j] == i:
                ii[cnt],jj[cnt],vv[cnt] = i,j,1
                cnt+=1



def compute_R_and_P_kmeans():
    print(">>Computing P and R...")
    t = time.perf_counter()

    from scipy.cluster.vq import vq, kmeans, whiten

    # ----------------------------------- kmans ---------------------------------- #
    print("kmeans start")
    input = edge_center.to_numpy()

    NCONS = NE
    global new_M
    print("NCONS: ", NCONS, "  new_M: ", new_M)

    # run kmeans
    input = whiten(input)
    print("whiten done")

    print("computing kmeans...")
    kmeans_centroids, distortion = kmeans(obs=input, k_or_guess=new_M, iter=5)
    labels, _ = vq(input, kmeans_centroids)

    print("distortion: ", distortion)
    print("kmeans done")

    # ----------------------------------- R and P --------------------------------- #
    # 将labels转换为R
    i_arr = np.zeros((NCONS), dtype=np.int32)
    j_arr = np.zeros((NCONS), dtype=np.int32)
    v_arr = np.zeros((NCONS), dtype=np.int32)
    compute_R_based_on_kmeans_label_triplets(labels, i_arr, j_arr, v_arr, new_M, NCONS)

    R = scipy.sparse.coo_array((v_arr, (i_arr, j_arr)), shape=(new_M, NCONS)).tocsr()
    P = R.transpose()
    print(f"Computing P and R done, time = {time.perf_counter() - t}")

    # print(f"writing P and R...")
    # scipy.io.mmwrite("R.mtx", R)
    # scipy.io.mmwrite("P.mtx", P)
    # print(f"writing P and R done")

    return R, P, labels, new_M

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
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraints[i] = dis.norm() - rest_len[i]
        g = dis.normalized()

        gradC[i, 0] = g
        gradC[i, 1] = -g


@ti.kernel
def compute_K_kernel(K_diag:ti.types.ndarray()):
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        L= dis.norm()
        g = dis.normalized()

        #geometric stiffness K: 
        # https://github.com/FantasyVR/magicMirror/blob/a1e56f79504afab8003c6dbccb7cd3c024062dd9/geometric_stiffness/meshComparison/meshgs_SchurComplement.py#L143
        # https://team.inria.fr/imagine/files/2015/05/final.pdf eq.21
        # https://blog.csdn.net/weixin_43940314/article/details/139448858
        k0 = lagrangian[i] / L * (1 - g[0]*g[0])
        k1 = lagrangian[i] / L * (1 - g[1]*g[1])
        k2 = lagrangian[i] / L * (1 - g[2]*g[2])
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
    for i in range(NE):
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
    for i in range(NE):
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

# https://github.com/pyamg/pyamg/blob/5a51432782c8f96f796d7ae35ecc48f81b194433/pyamg/relaxation/relaxation.py#L586
chebyshev_coeff = None
def chebyshev(A, x, b, coefficients, iterations=1):
    x = np.ravel(x)
    b = np.ravel(b)
    for _i in range(iterations):
        residual = b - A*x
        h = coefficients[0]*residual
        for c in coefficients[1:]:
            h = c*residual + A*h
        x += h

def calc_spectral_radius(A):
    global spectral_radius
    t = time.perf_counter()
    if args.use_cuda:
        cuda_set_A0(A)
        spectral_radius = extlib.fastmg_get_max_eig()
    else:
        spectral_radius = approximate_spectral_radius(A) # legacy python version
    print(f"spectral_radius time: {time.perf_counter()-t:.2f}s")
    print("spectral_radius:", spectral_radius)
    return spectral_radius


def setup_chebyshev(A, lower_bound=1.0/30.0, upper_bound=1.1, degree=3):
    global chebyshev_coeff 
    """Set up Chebyshev."""
    rho = calc_spectral_radius(A)
    a = rho * lower_bound
    b = rho * upper_bound
    chebyshev_coeff = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]


def setup_jacobi(A):
    from pyamg.relaxation.smoothing import rho_D_inv_A
    global jacobi_omega
    rho = rho_D_inv_A(A)
    print("rho:", rho)
    jacobi_omega = 1.0/(rho)
    print("omega:", jacobi_omega)


def build_Ps(A, method='UA'):
    """Build a list of prolongation matrices Ps from A """
    if method == 'UA':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, improve_candidates=None, symmetry='symmetric')
    elif method == 'SA' :
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400)
    elif method == 'CAMG':
        ml = pyamg.ruge_stuben_solver(A, max_coarse=400)
    else:
        raise ValueError(f"Method {method} not recognized")

    Ps = []
    for i in range(len(ml.levels)-1):
        Ps.append(ml.levels[i].P)

    return Ps


class MultiLevel:
    A = None
    P = None
    R = None


def build_levels(A, Ps=[]):
    '''Give A and a list of prolongation matrices Ps, return a list of levels'''
    lvl = len(Ps) + 1 # number of levels

    levels = [MultiLevel() for i in range(lvl)]

    levels[0].A = A

    for i in range(lvl-1):
        levels[i].P = Ps[i]
        levels[i].R = Ps[i].T
        levels[i+1].A = Ps[i].T @ levels[i].A @ Ps[i]

    return levels


def setup_smoothers(A):
    global chebyshev_coeff
    if args.smoother_type == 'chebyshev':
        setup_chebyshev(A, lower_bound=1.0/30.0, upper_bound=1.1, degree=3)
    elif args.smoother_type == 'jacobi':
        setup_jacobi(A)
        extlib.fastmg_setup_jacobi(jacobi_omega, 10)


def old_amg_cg_solve(levels, b, x0=None, tol=1e-5, maxiter=100):
    assert x0 is not None
    x = x0.copy()
    A = levels[0].A
    residuals = np.zeros(maxiter+1)
    def psolve(b):
        x = x0.copy()
        old_V_cycle(levels, 0, x, b)
        return x
    bnrm2 = np.linalg.norm(b)
    atol = tol * bnrm2
    r = b - A@(x)
    rho_prev, p = None, None
    normr = np.linalg.norm(r)
    residuals[0] = normr
    iteration = 0
    for iteration in range(maxiter):
        if normr < atol:  # Are we done?
            break
        z = psolve(r)
        rho_cur = np.dot(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]
        q = A@(p)
        alpha = rho_cur / np.dot(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur
        normr = np.linalg.norm(r)
        residuals[iteration+1] = normr
    residuals = residuals[:iteration+1]
    return (x),  residuals  

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


def diag_sweep(A,x,b,iterations=1):
    diag = A.diagonal()
    diag = np.where(diag==0, 1, diag)
    x[:] = b / diag

def presmoother(A,x,b):
    from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
    if args.smoother_type == 'gauss_seidel':
        gauss_seidel(A,x,b,iterations=1, sweep='symmetric')
    elif args.smoother_type == 'jacobi':
        jacobi(A,x,b,iterations=10, omega=jacobi_omega)
    elif args.smoother_type == 'sor_vanek':
        for _ in range(1):
            sor(A,x,b,omega=1.0,iterations=1,sweep='forward')
            sor(A,x,b,omega=1.85,iterations=1,sweep='backward')
    elif args.smoother_type == 'sor':
        sor(A,x,b,omega=1.33,sweep='symmetric',iterations=1)
    elif args.smoother_type == 'diag_sweep':
        diag_sweep(A,x,b,iterations=1)
    elif args.smoother_type == 'chebyshev':
        chebyshev(A,x,b,chebyshev_coeff)


def postsmoother(A,x,b):
    presmoother(A,x,b)


def coarse_solver(A, b):
    res = np.linalg.solve(A.toarray(), b)
    return res

def old_V_cycle(levels,lvl,x,b):
    A = levels[lvl].A.astype(np.float64)
    presmoother(A,x,b)
    residual = b - A @ x
    coarse_b = levels[lvl].R @ residual
    coarse_x = np.zeros_like(coarse_b)
    if lvl == len(levels)-2:
        coarse_x = coarse_solver(levels[lvl+1].A, coarse_b)
    else:
        old_V_cycle(levels, lvl+1, coarse_x, coarse_b)
    x += levels[lvl].P @ coarse_x
    postsmoother(A, x, b)



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
    lagrangian.from_numpy(lagrangian.to_numpy() + dLambda_)
    dpos = M_inv @ G.transpose() @ dLambda_ 
    if use_PXPBD_v1:
        dpos -=  Minv_gg
    dpos = dpos.reshape(-1, 3)
    pos.from_numpy(pos_mid.to_numpy() + omega*dpos)

@ti.kernel
def transfer_back_to_pos_mfree_kernel():
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]

        delta_lagrangian = dLambda[i]
        lagrangian[i] += delta_lagrangian

        gradient = gradC[i, 0]
        
        if invM0 != 0.0:
            dpos[idx0] += invM0 * delta_lagrangian * gradient
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient


@ti.kernel
def transfer_back_to_pos_mfree_kernel_withg():
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        gradient = gradC[i, 0]
        if invM0 != 0.0:
            dpos_withg[idx0] += invM0 * lagrangian[i] * gradient 
        if invM1 != 0.0:
            dpos_withg[idx1] -= invM1 * lagrangian[i] * gradient

    for i in range(NV):
        if inv_mass[i] != 0.0:
            dpos_withg[i] += predict_pos[i] - old_pos[i]


def transfer_back_to_pos_mfree(x):
    dLambda.from_numpy(x)
    reset_dpos(dpos)
    transfer_back_to_pos_mfree_kernel()
    update_pos(inv_mass, dpos, pos)

def spy_A(A,b):
    print("A:", A.shape, " b:", b.shape)
    scipy.io.mmwrite("A.mtx", A)
    plt.spy(A, markersize=1)
    plt.show()
    exit()


def fill_G():
    tic = time.perf_counter()
    compute_C_and_gradC_kernel(pos, gradC, edge, constraints, rest_len)
    G_ii, G_jj, G_vv = np.zeros(NCONS*6, dtype=np.int32), np.zeros(NCONS*6, dtype=np.int32), np.zeros(NCONS*6, dtype=np.float32)
    fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, gradC, edge)
    G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(NCONS, 3 * NV))
    print(f"    fill_G: {time.perf_counter() - tic:.4f}s")
    return G


# legacy
def fill_A_by_spmm(M_inv, ALPHA):
    tic = time.perf_counter()
    G_ii, G_jj, G_vv = np.zeros(NCONS*6, dtype=np.int32), np.zeros(NCONS*6, dtype=np.int32), np.zeros(NCONS*6, dtype=np.float32)
    fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, gradC, edge)
    G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(NCONS, 3 * NV))
    print(f"fill_G: {time.perf_counter() - tic:.4f}s")

    tic = time.perf_counter()
    if use_geometric_stiffness:
        # Geometric Stiffness: K = NCONS - H, we only use diagonal of H and then replace M_inv with K_inv
        # https://github.com/FantasyVR/magicMirror/blob/a1e56f79504afab8003c6dbccb7cd3c024062dd9/geometric_stiffness/meshComparison/meshgs_SchurComplement.py#L143
        # https://team.inria.fr/imagine/files/2015/05/final.pdf eq.21
        # https://blog.csdn.net/weixin_43940314/article/details/139448858
        K_diag.fill(0.0)
        compute_K_kernel(K_diag)
        mass = 1.0/(M_inv.diagonal()+1e-12)
        MK_inv = scipy.sparse.diags([1.0/(mass - K_diag)], [0], format="dia")
        M_inv = MK_inv # replace old M_inv with MK_inv

    A = G @ M_inv @ G.transpose() + ALPHA
    A = scipy.sparse.csr_matrix(A)
    # print("fill_A_by_spmm  time: ", time.perf_counter() - tic)
    return A, G




def calc_num_nonz(num_adjacent_edge):
    num_nonz = np.sum(num_adjacent_edge)+num_adjacent_edge.shape[0]
    return num_nonz

def calc_nnz_each_row(num_adjacent_edge):
    nnz_each_row = num_adjacent_edge[:] + 1
    return nnz_each_row

def init_A_CSR_pattern(num_adjacent_edge, adjacent_edge):
    num_adj = num_adjacent_edge
    adj = adjacent_edge
    nonz = np.sum(num_adj)+NE
    indptr = np.zeros(NE+1, dtype=np.int32)
    indices = np.zeros(nonz, dtype=np.int32)
    data = np.zeros(nonz, dtype=np.float32)

    indptr[0] = 0
    for i in range(0,NE):
        num_adj_i = num_adj[i]
        indptr[i+1]=indptr[i] + num_adj_i + 1
        indices[indptr[i]:indptr[i+1]-1]= adj[i][:num_adj_i]
        indices[indptr[i+1]-1]=i

    assert indptr[-1] == nonz

    return data, indices, indptr


def csr_index_to_coo_index(indptr, indices):
    ii, jj = np.zeros_like(indices), np.zeros_like(indices)
    for i in range(NE):
        ii[indptr[i]:indptr[i+1]]=i
        jj[indptr[i]:indptr[i+1]]=indices[indptr[i]:indptr[i+1]]
    return ii, jj


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
            data[cnt] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha
            continue
        a = adjacent_edge_abc[i, k * 3]
        b = adjacent_edge_abc[i, k * 3 + 1]
        c = adjacent_edge_abc[i, k * 3 + 2]
        g_ab = (pos[a] - pos[b]).normalized()
        g_ac = (pos[a] - pos[c]).normalized()
        offdiag = inv_mass[a] * g_ab.dot(g_ac)
        data[cnt] = offdiag

# For i and for k version
# Input is already in CSR format. We only update the data.
@ti.kernel
def fill_A_offdiag_CSR_2_kernel(data:ti.types.ndarray(dtype=ti.f32)):
    cnt = 0
    ti.loop_config(serialize=True)
    for i in range(NE):
        for k in range(num_adjacent_edge[i]):
            a = adjacent_edge_abc[i, k * 3]
            b = adjacent_edge_abc[i, k * 3 + 1]
            c = adjacent_edge_abc[i, k * 3 + 2]
            g_ab = (pos[a] - pos[b]).normalized()
            g_ac = (pos[a] - pos[c]).normalized()
            offdiag = inv_mass[a] * g_ab.dot(g_ac)
            data[cnt] = offdiag
            cnt += 1
        cnt += 1 # diag


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


def fill_A_ijv_ti():
    ii, jj, vv = np.zeros(num_nonz, int), np.zeros(num_nonz, int), np.zeros(num_nonz, np.float32)
    fill_A_ijv_kernel(ii, jj, vv, num_adjacent_edge, adjacent_edge, adjacent_edge_abc, inv_mass, alpha)
    A = scipy.sparse.coo_array((vv, (ii, jj)), shape=(NE, NE))
    A.eliminate_zeros()
    A= A.tocsr()


def fill_A_csr_ti():
    fill_A_CSR_kernel(spmat_data, spmat_indptr, spmat_ii, spmat_jj, adjacent_edge_abc, num_nonz, alpha)
    A = scipy.sparse.csr_matrix((spmat_data, spmat_indices, spmat_indptr), shape=(NE, NE))
    return A


def fill_A_mfree_wrapper():
    ii, jj, vv = np.zeros(2*num_nonz, int), np.zeros(2*num_nonz, int), np.zeros(2*num_nonz, np.float32)
    fill_A_mfree(v2e, num_v2e, ii, jj, vv, pos.to_numpy(), edge.to_numpy(), inv_mass.to_numpy())
    A = scipy.sparse.coo_array((vv, (ii, jj)), shape=(NE, NE))
    A= A.tocsr()
    return A

def normalize(v):
    return v / np.linalg.norm(v)

def compare_find_shared_v_order(v,e1,e2,edge):
    # which is shared v in e1? 0 or 1
    order_in_e1 = 0 if edge[e1][0] == v else 1
    order_in_e2 = 0 if edge[e2][0] == v else 1
    return order_in_e1, order_in_e2


def fill_A_mfree(v2e, num_v2e, ii, jj, vv, pos, edge, inv_mass):
    cnt=0
    for v in range(NV):
        es = v2e[v] # a list of edges
        if inv_mass[v] == 0: # no mass, no force
            continue
        if num_v2e[v] == 0: #only one edge, no shared edge
            continue
        else:
            for i in range(num_v2e[v]):
                for j in range(num_v2e[v]):
                    if i == j:
                        continue
                    e1 = es[i]
                    e2 = es[j]

                    o1,o2 = compare_find_shared_v_order(v,e1,e2,edge)
                    o1 = 1-o1 # 0->1, 1->0, because we want to find the other point
                    o2 = 1-o2

                    g1 = normalize(pos[v] - pos[edge[e1][o1]])
                    g2 = normalize(pos[v] - pos[edge[e2][o2]])

                    ii[cnt]=e1
                    jj[cnt]=e2 # A[e1, e2]
                    vv[cnt] = inv_mass[v] * g1.dot(g2) 
                    cnt+=1

    # diagonal
    for i in range(NE):
        ii[cnt]=i
        jj[cnt]=i
        vv[cnt]=inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha
        cnt+=1




def fastFill_fetch():
    global spmat_data, spmat_indices, spmat_indptr
    extlib.fastFillCloth_fetch_A_data(spmat_data)
    A = scipy.sparse.csr_matrix((spmat_data, spmat_indices, spmat_indptr), shape=(NE, NE))
    return A


def fastFillCloth_run():
    extlib.fastFillCloth_run(pos.to_numpy())


@ti.kernel
def fill_A_diag_kernel(diags:ti.types.ndarray(dtype=ti.f32), alpha:ti.f32, inv_mass:ti.template(), edge:ti.template()):
    for i in range(edge.shape[0]):
        diags[i] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha


@ti.kernel
def fill_A_ijv_kernel(ii:ti.types.ndarray(dtype=ti.i32), jj:ti.types.ndarray(dtype=ti.i32), vv:ti.types.ndarray(dtype=ti.f32), num_adjacent_edge:ti.types.ndarray(dtype=ti.i32), adjacent_edge:ti.types.ndarray(dtype=ti.i32), adjacent_edge_abc:ti.types.ndarray(dtype=ti.i32),  inv_mass:ti.template(), alpha:ti.f32):
    n = 0
    NE = adjacent_edge.shape[0]
    ti.loop_config(serialize=True)
    for i in range(NE): #对每个edge，找到所有的adjacent edge，填充到offdiag，然后填充diag
        for k in range(num_adjacent_edge[i]):
            ia = adjacent_edge[i,k]
            a = adjacent_edge_abc[i, k * 3]
            b = adjacent_edge_abc[i, k * 3 + 1]
            c = adjacent_edge_abc[i, k * 3 + 2]
            g_ab = (pos[a] - pos[b]).normalized()
            g_ac = (pos[a] - pos[c]).normalized()
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
        vv[n] = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha
        n += 1 


def export_A_b(A,b,postfix="", binary=args.export_matrix_binary):
    tic = time.perf_counter()
    dir = out_dir + "/A/"
    if binary:
        # https://stackoverflow.com/a/8980156/19253199
        scipy.sparse.save_npz(dir + f"A_{postfix}.npz", A)
        np.save(dir + f"b_{postfix}.npy", b)
        # A = scipy.sparse.load_npz("A.npz") # load
        # b = np.load("b.npy")
    else:
        scipy.io.mmwrite(dir + f"A_{postfix}.mtx", A, symmetry='symmetric')
        np.savetxt(dir + f"b_{postfix}.txt", b)
    t_calc_residual += time.perf_counter()-tic
    print(f"    export_A_b time: {time.perf_counter()-tic:.3f}s")
    

@ti.kernel
def compute_potential_energy():
    potential_energy[None] = 0.0
    inv_alpha = 1.0/compliance
    for i in range(NE):
        potential_energy[None] += 0.5 * inv_alpha * constraints[i]**2

@ti.kernel
def compute_inertial_energy():
    inertial_energy[None] = 0.0
    inv_h2 = 1.0 / delta_t**2
    for i in range(NV):
        if inv_mass[i] == 0.0:
            continue
        inertial_energy[None] += 0.5 / inv_mass[i] * (pos[i] - predict_pos[i]).norm_sqr() * inv_h2


def AMG_python(b):
    global Ps, num_levels

    A = fill_A_csr_ti()

    if should_setup():
        tic = time.perf_counter()
        Ps = build_Ps(A)
        num_levels = len(Ps)+1
        logging.info(f"    build_Ps time:{time.perf_counter()-tic}")
    
    tic = time.perf_counter()
    levels = build_levels(A, Ps)
    logging.info(f"    build_levels time:{time.perf_counter()-tic}")

    if should_setup():
        tic = time.perf_counter()
        setup_smoothers(A)
        logging.info(f"    setup smoothers time:{perf_counter()-tic}")
    x0 = np.zeros_like(b)
    tic = time.perf_counter()
    x, r_Axb = old_amg_cg_solve(levels, b, x0=x0, maxiter=args.maxiter_Axb, tol=1e-6)
    toc = time.perf_counter()
    logging.info(f"    mgsolve time {toc-tic}")
    return  x, r_Axb


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
    return ((frame%args.setup_interval==0 or (args.restart==True and frame==args.restart_frame)) and (ite==0))



def AMG_calc_r(r, r0, tic_iter, r_Axb):
    global t_export
    tic = time.perf_counter()

    t_iter = perf_counter()-tic_iter
    tic_calcr = perf_counter()
    calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
    dual_r = np.linalg.norm(dual_residual.to_numpy()).astype(float)
    # compute_potential_energy()
    # compute_inertial_energy()
    # robj = (potential_energy[None]+inertial_energy[None])
    r_Axb = r_Axb.tolist()
    if use_PXPBD_v1:
        G = fill_G()
        primary_residual = calc_primary_residual(G, M_inv)
        primal_r = np.linalg.norm(primary_residual).astype(float)
        Newton_r = np.sqrt(primal_r**2 + dual_r**2)

    # if export_fullr:
    #     fulldual_final = dual_residual.to_numpy()
    #     np.savez_compressed(out_dir+'/r/'+ f'fulldual_{frame}-{ite}', fulldual0, fulldual_final)

    logging.info(f"    convergence factor: {calc_conv(r_Axb):.2f}")
    logging.info(f"    Calc r time: {(perf_counter()-tic_calcr)*1000:.0f}ms")

    if args.export_log:
        logging.info(f"    iter total time: {t_iter*1000:.0f}ms")
        if use_PXPBD_v1:
            r.append(ResidualDataPrimal(dual_r, primal_r, Newton_r, len(r_Axb), t_iter))
            logging.info(f"{frame}-{ite} rsys:{r_Axb[0]:.2e} {r_Axb[-1]:.2e} r0:{r0.Newton:.2e} dual:{dual_r:.2e} primal:{primal_r:.2e} Newton:{Newton_r:.2e} iter:{len(r_Axb)}")
        else:
            r.append(ResidualData(dual_r, len(r_Axb), t_iter))
            logging.info(f"{frame}-{ite} rsys:{r_Axb[0]:.2e} {r_Axb[-1]:.2e} r0:{r0.dual:.2e} dual:{dual_r:.2e}  iter:{len(r_Axb)}")

    t_export += perf_counter()-tic



def AMG_calc_r0():
    global t_export
    calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
    dual_r = np.linalg.norm(dual_residual.to_numpy()).astype(float)
    # compute_potential_energy()
    # compute_inertial_energy()
    # robj = (potential_energy[None]+inertial_energy[None])
    if use_PXPBD_v1:
        G = fill_G()
        primary_residual = calc_primary_residual(G, M_inv)
        primal_r = np.linalg.norm(primary_residual).astype(float)
        Newton_r = np.sqrt(primal_r**2 + dual_r**2)

        r0 = (Residual0DataPrimal(dual_r, primal_r, Newton_r))
    else:
        r0 = (Residual0Data(dual_r))
    return r0


def do_export_r(r):
    global t_export
    tic = time.perf_counter()
    serialized_r = [r[i]._asdict() for i in range(len(r))]
    r_json = json.dumps(serialized_r)
    with open(out_dir+'/r/'+ f'{frame}.json', 'w') as file:
        file.write(r_json)
    t_export += time.perf_counter()-tic


@ti.kernel
def PXPBD_b_kernel(pos:ti.template(), predict_pos:ti.template(), lagrangian:ti.template(), inv_mass:ti.template(), gradC:ti.template(), b:ti.types.ndarray(), Minv_gg:ti.template()):
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]

        if invM0 != 0.0:
            Minv_gg[idx0] = invM0 * lagrangian[i] * gradC[i, 0] + (pos[idx0] - predict_pos[idx0])
        if invM1 != 0.0:
            Minv_gg[idx1] = invM1 * lagrangian[i] * gradC[i, 1] + (pos[idx0] - predict_pos[idx0])

    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        if invM1 != 0.0 and invM0 != 0.0:
            b[idx0] += gradC[i, 0] @ Minv_gg[idx0] + gradC[i, 1] @ Minv_gg[idx1]

        #     Minv_gg =  (pos.to_numpy().flatten() - predict_pos.to_numpy().flatten()) - M_inv @ G.transpose() @ lagrangian.to_numpy()
        #     b += G @ Minv_gg


# v1-mfree
def PXPBD_v1_mfree_transfer_back_to_pos(x, Minv_gg):
    dLambda.from_numpy(x)
    reset_dpos(dpos)
    PXPBD_v1_mfree_transfer_back_to_pos_kernel(Minv_gg)
    update_pos(inv_mass, dpos, pos)


# v1-mfree
@ti.kernel
def PXPBD_v1_mfree_transfer_back_to_pos_kernel(Minv_gg:ti.template()):
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]

        delta_lagrangian = dLambda[i]
        lagrangian[i] += delta_lagrangian

        gradient = gradC[i, 0]
        
        if invM0 != 0.0:
            dpos[idx0] += invM0 * delta_lagrangian * gradient - Minv_gg[idx0]
        if invM1 != 0.0:
            dpos[idx1] -= invM1 * delta_lagrangian * gradient - Minv_gg[idx1]
        


def AMG_setup_phase():
    global Ps, num_levels
    tic = time.perf_counter()
    # A = fill_A_csr_ti() taichi version
    A = fastFill_fetch()
    Ps = build_Ps(A)
    num_levels = len(Ps)+1
    logging.info(f"    build_Ps time:{time.perf_counter()-tic}")

    extlib.fastmg_setup_nl(num_levels)

    tic = time.perf_counter()
    update_P(Ps)
    logging.info(f"    update_P time: {time.perf_counter()-tic:.2f}s")

    tic = time.perf_counter()
    cuda_set_A0(A)
    extlib.fastmg_setup_smoothers(1) # 1 means chebyshev
    logging.info(f"    setup smoothers time:{perf_counter()-tic}")

    report_multilevel_details(Ps, num_levels)


def AMG_RAP():
    tic3 = time.perf_counter()
    for lv in range(num_levels-1):
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
    lagrangian.from_numpy(lagrangian.to_numpy() + dLambda_)
    dpos = M_inv @ G.transpose() @ dLambda_ 
    dpos -=  Minv_gg
    dpos = dpos.reshape(-1, 3)
    pos.from_numpy(pos_mid.to_numpy() + omega*dpos)


# v2: blended, only modify dpos
def AMG_PXPBD_v2_dlam2dpos(x):
    tic = time.perf_counter()
    dLambda.from_numpy(x)
    reset_dpos(dpos)
    transfer_back_to_pos_mfree_kernel()
    transfer_back_to_pos_mfree_kernel_withg()
    update_pos_blend(inv_mass, dpos, pos, dpos_withg)
    logging.info(f"    dlam2dpos time: {(perf_counter()-tic)*1000:.0f}ms")


def AMG_b():
    update_constraints_kernel(pos, edge, rest_len, constraints)
    b = -constraints.to_numpy() - alpha_tilde_np * lagrangian.to_numpy()
    return b


def AMG_PXPBD_v1_b(G):
    # #we calc inverse mass times gg(primary residual), because NCONS may contains infinity for fixed pin points. And gg always appears with inv_mass.
    update_constraints_kernel(pos, edge, rest_len, constraints)
    b = -constraints.to_numpy() - alpha_tilde_np * lagrangian.to_numpy()

    # PXPBD_b_kernel(pos, predict_pos, lagrangian, inv_mass, gradC, b, Minv_gg)
    Minv_gg =  (pos.to_numpy().flatten() - predict_pos.to_numpy().flatten()) - M_inv @ G.transpose() @ lagrangian.to_numpy()
    b += G @ Minv_gg
    return b, Minv_gg
    

def AMG_A():
    tic2 = perf_counter()
    fastFillCloth_run()
    logging.info(f"    fill_A time: {(perf_counter()-tic2)*1000:.0f}ms")

def calc_dual():
    calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
    return dual_residual.to_numpy()


def substep_all_solver():
    global ite, t_export, n_outer_all
    tic1 = time.perf_counter()
    semi_euler(old_pos, inv_mass, vel, pos)
    reset_lagrangian(lagrangian)
    r = [] # residual list of one frame
    # fulldual0 = calc_dual()
    # print("dual0: ", np.linalg.norm(fulldual0))
    logging.info(f"pre-loop time: {(perf_counter()-tic1)*1000:.0f}ms")
    r0 = AMG_calc_r0()
    for ite in range(args.maxiter):
        tic_iter = perf_counter()
        if use_PXPBD_v1:
            copy_field(pos_mid, pos)
        compute_C_and_gradC_kernel(pos, gradC, edge, constraints, rest_len) # required by dlam2dpos
        b = AMG_b()
        if use_PXPBD_v1:
            G = fill_G()
            b, Minv_gg = AMG_PXPBD_v1_b(G)
        if not args.use_cuda:
            x, r_Axb = AMG_python(b)
        else:
            AMG_A()
            if should_setup():
                AMG_setup_phase()
            extlib.fastmg_set_A0_from_fastFillCloth()
            AMG_RAP()
            x, r_Axb = AMG_solve(b, maxiter=args.maxiter_Axb, tol=1e-5)
        if use_PXPBD_v1:
            AMG_PXPBD_v1_dlam2dpos(x, G, Minv_gg)
        else:
            AMG_dlam2dpos(x)
        AMG_calc_r(r, r0, tic_iter, r_Axb)
        logging.info(f"iter time(with export): {(perf_counter()-tic_iter)*1000:.0f}ms")

        if use_PXPBD_v1:
            if r[-1].Newton < 0.1*r0.Newton or r[-1].Newton<1e-5:
                break
        else:
            if r[-1].dual < 0.1*r0.dual or r[-1].dual<1e-5:
                break
    
    tic = time.perf_counter()
    logging.info(f"n_outer: {ite+1}")
    n_outer_all.append(ite+1)
    if args.export_residual:
        do_export_r(r)
    update_vel(old_pos, inv_mass, vel, pos)
    logging.info(f"post-loop time: {(time.perf_counter()-tic)*1000:.0f}ms")


def mkdir_if_not_exist(path=None):
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory_path):
        os.makedirs(path)

def delete_txt_files(folder_path):
    txt_files = glob.glob(os.path.join(folder_path, '*r_frame_*.txt'))
    for file_path in txt_files:
        os.remove(file_path)

def clean_result_dir(folder_path):
    from pathlib import Path
    pwd = os.getcwd()
    os.chdir(folder_path)
    print(f"clean {folder_path}...")
    except_files = ["b0.txt"]
    to_remove = []
    for wildcard_name in [
        '*.obj',
        '*.png',
        '*.ply',
        '*.txt',
        '*.json',
        '*.npz',
        '*.mtx',
        '*.log'
    ]:
        files = glob.glob(wildcard_name)
        to_remove += (files)
        for f in files:
            if f in except_files:
                to_remove.remove(f)
    logging.info(f"removing {len(to_remove)} files")
    for file_path in to_remove:
        os.remove(file_path)
    logging.info(f"clean {folder_path} done")
    os.chdir(pwd)

def use_another_outdir(out_dir):
    path = Path(out_dir)
    if path.exists():
        # add a number to the end of the folder name
        path = path.parent / (path.name + "_1")
        if path.exists():
            i = 2
            while True:
                path = path.parent / (path.name[:-2] + f"_{i}")
                if not path.exists():
                    break
                i += 1
    # path.mkdir(parents=True, exist_ok=True)
    out_dir = str(path)
    return out_dir

@ti.kernel
def copy_field(dst: ti.template(), src: ti.template()):
    for i in src:
        dst[i] = src[i]


def write_mesh(filename, pos, tri, format="ply"):
    cells = [
        ("triangle", tri.reshape(-1, 3)),
    ]
    mesh = meshio.Mesh(
        pos,
        cells,
    )

    if format == "ply":
        mesh.write(filename + ".ply", binary=True)
    elif format == "obj":
        mesh.write(filename + ".obj")
    else:
        raise ValueError("Unknown format")
    return mesh

@ti.kernel
def init_scale():
    scale = 1.5
    for i in range(NV):
        pos[i] *= scale


def save_state(filename):
    global frame, pos, vel, old_pos, predict_pos
    state = [frame, pos, vel, old_pos, predict_pos, rest_len]
    for i in range(1, len(state)):
        state[i] = state[i].to_numpy()
    np.savez(filename, *state)
    print(f"Saved frame-{frame} states to '{filename}', {len(state)} variables")

def load_state(filename):
    global frame, pos, vel, old_pos, predict_pos
    npzfile = np.load(filename)
    state = [frame, pos, vel, old_pos, predict_pos, rest_len]
    frame = int(npzfile["arr_0"])
    for i in range(1, len(state)):
        state[i].from_numpy(npzfile["arr_" + str(i)])
    print(f"Loaded frame-{frame} states to '{filename}', {len(state)} variables")


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


def find_last_frame(dir):
    # find the last frame number of dir
    files = glob.glob(dir + "/state/*.npz")
    files.sort(key=os.path.getmtime)
    if len(files) == 0:
        return 0
    path = Path(files[-1])
    last_frame = int(path.stem)
    return last_frame


def dict_to_ndarr(d:dict)->np.ndarray:
    lengths = np.array([len(v) for v in d.values()])

    max_len = max(len(item) for item in d.values())
    # 使用填充或截断的方式转换为NumPy数组
    arr = np.array([list(item) + [-1]*(max_len - len(item)) if len(item) < max_len else list(item)[:max_len] for item in d.values()])
    return arr, lengths



def init_set_P_R_manually():
    misc_dir_path = prj_path + "/data/misc/"
    if args.solver_type=="AMG":
        # init_edge_center(edge_center, edge, pos)
        if save_P:
            R, P, labels, new_M = compute_R_and_P_kmeans()
            scipy.io.mmwrite(misc_dir_path + "R.mtx", R)
            scipy.io.mmwrite(misc_dir_path + "P.mtx", P)
            np.savetxt(misc_dir_path + "labels.txt", labels, fmt="%d")
        if load_P:
            R = scipy.io.mmread(misc_dir_path+ "R.mtx")
            P = scipy.io.mmread(misc_dir_path+ "P.mtx")
            # labels = np.loadtxt( "labels.txt", dtype=np.int32)


def initFill_python():
    tic1 = perf_counter()
    print("Initializing adjacent edge and abc...")
    adjacent_edge, v2e_dict = init_adj_edge(edges=edge.to_numpy())
    adjacent_edge,num_adjacent_edge = dict_to_ndarr(adjacent_edge)
    v2e_np, num_v2e = dict_to_ndarr(v2e_dict)

    adjacent_edge_abc = np.empty((NE, 20*3), dtype=np.int32)
    adjacent_edge_abc.fill(-1)
    init_adjacent_edge_abc_kernel(NE,edge,adjacent_edge,num_adjacent_edge,adjacent_edge_abc)

    num_nonz = calc_num_nonz(num_adjacent_edge) 
    data, indices, indptr = init_A_CSR_pattern(num_adjacent_edge, adjacent_edge)
    ii, jj = csr_index_to_coo_index(indptr, indices)
    print(f"initFill time: {perf_counter()-tic1:.3f}s")
    return adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, data, indices, indptr, ii, jj, v2e_np, num_v2e


def initFill_cpp():
    tic1 = perf_counter()
    print("Initializing adjacent edge and abc...")
    extlib.initFillCloth_set(edge.to_numpy(), NE)
    extlib.initFillCloth_run()
    num_nonz = extlib.initFillCloth_get_nnz()

    MAX_ADJ = 20
    MAX_V2E = MAX_ADJ
    adjacent_edge = np.zeros((NE, MAX_ADJ), dtype=np.int32)
    num_adjacent_edge = np.zeros(NE, dtype=np.int32)
    adjacent_edge_abc = np.zeros((NE, MAX_ADJ*3), dtype=np.int32)
    spmat_data = np.zeros(num_nonz, dtype=np.float32)
    spmat_indices = np.zeros(num_nonz, dtype=np.int32)
    spmat_indptr = np.zeros(NE+1, dtype=np.int32)
    spmat_ii = np.zeros(num_nonz, dtype=np.int32)
    spmat_jj = np.zeros(num_nonz, dtype=np.int32)
    v2e = np.zeros((NV, MAX_V2E), dtype=np.int32)
    num_v2e = np.zeros(NV, dtype=np.int32)

    extlib.initFillCloth_get(adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, spmat_indices, spmat_indptr, spmat_ii, spmat_jj, v2e, num_v2e)
    print(f"initFill time: {perf_counter()-tic1:.3f}s")
    return adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, spmat_data, spmat_indices, spmat_indptr, spmat_ii, spmat_jj, v2e, num_v2e


def cache_and_initFill():
    global adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, spmat_data, spmat_indices, spmat_indptr, spmat_ii, spmat_jj, v2e, num_v2e
    if  os.path.exists(f'cache_initFill_N{N}.npz') and args.use_cache:
        npzfile= np.load(f'cache_initFill_N{N}.npz')
        (adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, spmat_data, spmat_indices, spmat_indptr, spmat_ii, spmat_jj) = (npzfile[key] for key in ['adjacent_edge', 'num_adjacent_edge', 'adjacent_edge_abc', 'num_nonz', 'spmat_data', 'spmat_indices', 'spmat_indptr', 'spmat_ii', 'spmat_jj'])
        num_nonz = int(num_nonz) # npz save int as np.array, it will cause bug in taichi kernel
        print(f"load cache_initFill_N{N}.npz")
    else:
        if args.use_cuda and use_cpp_initFill:
            initFill = initFill_cpp
        else:
            initFill = initFill_python
        adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, spmat_data, spmat_indices, spmat_indptr, spmat_ii, spmat_jj, v2e, num_v2e = initFill()
        print("caching init fill...")
        tic = perf_counter() # savez_compressed will save 10x space(1.4G->140MB), but much slower(33s)
        np.savez(f'cache_initFill_N{N}.npz', adjacent_edge=adjacent_edge, num_adjacent_edge=num_adjacent_edge, adjacent_edge_abc=adjacent_edge_abc, num_nonz=num_nonz, spmat_data=spmat_data, spmat_indices=spmat_indices, spmat_indptr=spmat_indptr, spmat_ii=spmat_ii, spmat_jj=spmat_jj)
        print("time of caching:", perf_counter()-tic)


def load_cache_initFill_to_cuda():
    global adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, spmat_data, spmat_indices, spmat_indptr, spmat_ii, spmat_jj, v2e, num_v2e
    cache_and_initFill()
    extlib.fastFillCloth_set_data(edge.to_numpy(), NE, inv_mass.to_numpy(), NV, pos.to_numpy(), alpha)
    extlib.fastFillCloth_init_from_python_cache(adjacent_edge,
                                           num_adjacent_edge,
                                           adjacent_edge_abc,
                                           num_nonz,
                                           spmat_data,
                                           spmat_indices,
                                           spmat_indptr,
                                           spmat_ii,
                                           spmat_jj,
                                           NE,
                                           NV)


def ending(timer_loop, start_date, initial_frame, t_export_total):
    global n_outer_all
    t_all = time.perf_counter() - timer_loop
    end_date = datetime.datetime.now()
    args.end_frame = frame

    sum_n_outer = sum(n_outer_all)
    avg_n_outer = sum_n_outer / len(n_outer_all)
    max_n_outer = max(n_outer_all)
    max_n_outer_index = n_outer_all.index(max_n_outer)

    n_outer_all_np = np.array(n_outer_all, np.int32)    
    np.savetxt(out_dir+"/r/n_outer.txt", n_outer_all_np, fmt="%d")

    sim_cost_time = time.perf_counter() - timer_loop

    s = f"\n-------\n"+\
    f"Time: {(sim_cost_time):.2f}s = {(sim_cost_time)/60:.2f}min.\n" + \
    f"Frame {initial_frame}-{args.end_frame}({args.end_frame-initial_frame} frames)."+\
    f"\nAvg: {t_all/(args.end_frame-initial_frame):.2f}s/frame."+\
    f"\nStart\t{start_date},\nEnd\t{end_date}."+\
    f"\nTime of exporting: {t_export_total:.3f}s" + \
    f"\nSum n_outer: {sum_n_outer} \nAvg n_outer: {avg_n_outer:.1f}"+\
    f"\nMax n_outer: {max_n_outer} \nMax n_outer frame: {max_n_outer_index + initial_frame}." + \
    f"\nCloth-N{N}" + \
    f"\ndt={delta_t}" + \
    f"\nSolver: {args.solver_type}" + \
    f"\nout_dir: {out_dir}" 

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"result/meta/{current_time}.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(s)

    logging.info(s)




# ---------------------------------------------------------------------------- #
#                                initialization                                #
# ---------------------------------------------------------------------------- #



def do_restart():
    global frame
    if args.restart_from_last_frame :
        args.restart_frame =  find_last_frame(out_dir)
    if args.restart_frame == 0:
        print("No restart file found.")
    else:
        load_state(args.restart_dir + f"{args.restart_frame:04d}.npz")
        frame = args.restart_frame
        print(f"restart from last frame: {args.restart_frame}")


def make_and_clean_dirs(out_dir):
    import shutil
    from pathlib import Path
    global prj_path

    shutil.rmtree(out_dir, ignore_errors=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir + "/r/").mkdir(parents=True, exist_ok=True)
    Path(out_dir + "/A/").mkdir(parents=True, exist_ok=True)
    Path(out_dir + "/state/").mkdir(parents=True, exist_ok=True)
    Path(out_dir + "/mesh/").mkdir(parents=True, exist_ok=True)
    Path(prj_path + "/result/meta/").mkdir(parents=True, exist_ok=True)


def process_dirs():
    global out_dir
    if args.auto_another_outdir:
        out_dir = use_another_outdir(out_dir)
    make_and_clean_dirs(out_dir)


def init():
    global start_wall_time, frame, global_vars, tic_all
    process_dirs()
    
    log_level = logging.INFO
    if not args.export_log:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format="%(message)s",filename=out_dir + f'/latest.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"out_dir: {out_dir}")

    tic_all = time.perf_counter()
    start_wall_time = datetime.datetime.now()
    logging.info(f"start wall time:{start_wall_time}")

    print_all_globals(global_vars)


    logging.info("\nInitializing...")
    logging.info("Initializing pos..")
    init_pos(inv_mass,pos)
    init_tri(tri)
    init_edge(edge, rest_len, pos)
    if args.setup_num == 1:
        init_scale()
    write_mesh(out_dir + f"/mesh/{frame:04d}", pos.to_numpy(), tri.to_numpy())
    logging.info("Initializing pos and edge done")

    tic = time.perf_counter()
    if args.solver_type == "AMG":
        if args.use_cuda:
            load_cache_initFill_to_cuda()
        else:
            cache_and_initFill()
    logging.info(f"Init fill time: {time.perf_counter()-tic:.3f}s")

    if args.restart:
        do_restart()

    global M_inv, ALPHA, inv_mass_np, alpha_tilde_np
    inv_mass_np = np.repeat(inv_mass.to_numpy(), 3, axis=0)
    M_inv = scipy.sparse.diags(inv_mass_np)
    alpha_tilde_np = np.array([alpha] * NCONS)
    ALPHA = scipy.sparse.diags(alpha_tilde_np)

    logging.info(f"Initialization done. Cost time:  {time.perf_counter() - tic_all:.3f}s") 


def export_after_substep(tic_frame, t_export, t_export_total):
    tic_export = time.perf_counter()
    if export_mesh:
        write_mesh(out_dir + f"/mesh/{frame:04d}", pos.to_numpy(), tri.to_numpy())
    if args.export_state:
        save_state(out_dir+'/state/' + f"{frame:04d}.npz")
    t_export += time.perf_counter()-tic_export
    t_export_total += t_export
    t_frame = time.perf_counter()-tic_frame
    if args.export_log:
        logging.info(f"Time of exporting: {t_export:.3f}s")
        logging.info(f"Time of frame-{frame}: {t_frame:.3f}s")

class Viewer:
    if use_viewer:
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
        if use_viewer:
            viewer.camera.track_user_inputs(viewer.window, movement_speed=0.003, hold_key=ti.ui.RMB)
            viewer.scene.set_camera(viewer.camera)
            viewer.scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
            viewer.scene.mesh(pos, tri, color=(1.0,0,0), two_sided=True)
            viewer.canvas.scene(viewer.scene)
            # you must call this function, even if we just want to save the image, otherwise the GUI image will not update.
            viewer.window.show()
            if save_image:
                file_path = out_dir + f"{frame:04d}.png"
                viewer.window.save_image(file_path)  # export and show in GUI
        
    def do_render_control(viewer):
        if use_viewer:
            for e in viewer.window.get_events(ti.ui.PRESS):
                if e.key in [ti.ui.ESCAPE]:
                    exit()
                if e.key == ti.ui.SPACE:
                    paused = not paused
                    print("paused:",paused)

viewer = Viewer()


def run():
    global frame, paused, ite, t_export
    timer_loop = time.perf_counter()
    initial_frame = frame
    step_pbar = tqdm.tqdm(total=args.end_frame, initial=frame)
    t_export_total = 0.0
    try:
        while True:
            tic_frame = time.perf_counter()
            t_export = 0.0

            viewer.do_render_control()

            if not paused:
                if args.solver_type == "XPBD":
                    substep_xpbd()
                else:
                    substep_all_solver()
                frame += 1
                
                export_after_substep(tic_frame, t_export, t_export_total)

            if frame == args.end_frame:
                print("Normallly end.")
                ending(timer_loop, start_wall_time, initial_frame, t_export_total)
                exit()

            if use_viewer:
                viewer.do_render_taichi(viewer)

            logging.info("\n")
            step_pbar.update(1)
            logging.info("")

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        ending(timer_loop, start_wall_time, initial_frame, t_export_total)


def main():
    init()
    run()

if __name__ == "__main__":
    main()
