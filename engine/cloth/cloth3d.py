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
from pyamg.relaxation.relaxation import polynomial
from time import perf_counter
from scipy.linalg import pinv
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
use_offdiag = True
reduce_offdiag = False
early_stop = True
use_primary_residual = False
use_geometric_stiffness = False
dont_clean_results = False
report_time = True
# chebyshev_coeff = None
export_fullr = False
num_levels = 0
t_export = 0.0


#parse arguments to change default values
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1024)
parser.add_argument("-delta_t", type=float, default=1e-3)
parser.add_argument("-solver_type", type=str, default='AMG', help='"AMG", "GS", "XPBD"')
parser.add_argument("-export_matrix", type=int, default=False)
parser.add_argument("-export_matrix_binary", type=int, default=True)
parser.add_argument("-export_state", type=int, default=False)
parser.add_argument("-export_residual", type=int, default=False)
parser.add_argument("-end_frame", type=int, default=100)
parser.add_argument("-out_dir", type=str, default=f"result/latest/")
parser.add_argument("-auto_another_outdir", type=int, default=False)
parser.add_argument("-restart", type=int, default=False)
parser.add_argument("-restart_frame", type=int, default=10)
parser.add_argument("-restart_dir", type=str, default="result/latest/state/")
parser.add_argument("-restart_from_last_frame", type=int, default=True)
parser.add_argument("-max_iter", type=int, default=1000)
parser.add_argument("-max_iter_Axb", type=int, default=100)
parser.add_argument("-export_log", type=int, default=True)
parser.add_argument("-setup_num", type=int, default=0, help="attach:0, scale:1")
parser.add_argument("-use_json", type=int, default=False, help="json configs will overwrite the command line args")
parser.add_argument("-json_path", type=str, default="data/scene/cloth/config.json", help="json configs will overwrite the command line args")
parser.add_argument("-arch", type=str, default="cpu", help="taichi arch: cuda or cpu")
parser.add_argument("-use_cuda", type=int, default=True)
parser.add_argument("-cuda_dir", type=str, default="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")
parser.add_argument("-smoother_type", type=str, default="chebyshev")
parser.add_argument("-use_cache", type=int, default=True)
parser.add_argument("-use_fastFill", type=int, default=False)
parser.add_argument("-setup_interval", type=int, default=20)


args = parser.parse_args()
N = args.N
delta_t = args.delta_t
solver_type = args.solver_type
gravity = [0.0, -9.8, 0.0]
if args.setup_num==1: gravity = [0.0, 0.0, 0.0]
else : gravity = [0.0, -9.8, 0.0]
end_frame = args.end_frame
max_iter = args.max_iter
max_iter_Axb = args.max_iter_Axb
out_dir =  args.out_dir + "/"
smoother_type = args.smoother_type


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



if args.arch == "cuda":
    ti.init(arch=ti.cuda)
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
dpos     = ti.Vector.field(3, dtype=float, shape=(NV))
old_pos     = ti.Vector.field(3, dtype=float, shape=(NV))
vel         = ti.Vector.field(3, dtype=float, shape=(NV))
pos_mid     = ti.Vector.field(3, dtype=float, shape=(NV))
inv_mass    = ti.field(dtype=float, shape=(NV))
rest_len    = ti.field(dtype=float, shape=(NE))
lagrangian  = ti.field(dtype=float, shape=(NE))  
constraints = ti.field(dtype=float, shape=(NE))  
dLambda     = ti.field(dtype=float, shape=(NE))
numerator   = ti.field(dtype=float, shape=(NE))
denominator = ti.field(dtype=float, shape=(NE))
gradC       = ti.Vector.field(3, dtype = ti.float32, shape=(NE,2)) 
edge_center = ti.Vector.field(3, dtype = ti.float32, shape=(NE))
dual_residual       = ti.field(shape=(NE),    dtype = ti.float32) # -C - alpha * lagrangian
# adjacent_edge = ti.field(dtype=int, shape=(NE, 20))
# num_adjacent_edge = ti.field(dtype=int, shape=(NE))
# adjacent_edge_abc = ti.field(dtype=int, shape=(NE, 100))
# num_nonz = 0
nnz_each_row = np.zeros(NE, dtype=int)
potential_energy = ti.field(dtype=float, shape=())
inertial_energy = ti.field(dtype=float, shape=())
predict_pos = ti.Vector.field(3, dtype=float, shape=(NV))
# primary_residual = np.zeros(dtype=float, shape=(3*NV))
# K = ti.Matrix.field(3, 3, float, (NV, NV)) 
# geometric stiffness, only retain diagonal elements
K_diag = np.zeros((NV*3), dtype=float)

inv_mass_np = np.repeat(inv_mass.to_numpy(), 3, axis=0)
M_inv = scipy.sparse.diags(inv_mass_np)
alpha_tilde_np = np.array([alpha] * NCONS)
ALPHA = scipy.sparse.diags(alpha_tilde_np)


def init_extlib_argtypes():
    global extlib

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
    # extlib.fastmg_setup_chebyshev.argtypes = [arr_float, c_size_t]
    extlib.fastmg_setup_jacobi.argtypes = [ctypes.c_float, ctypes.c_size_t]
    extlib.fastmg_RAP.argtypes = [ctypes.c_size_t]
    extlib.fastmg_set_A0.argtypes = argtypes_of_csr
    extlib.fastmg_set_P.argtypes = [ctypes.c_size_t] + argtypes_of_csr
    extlib.fastmg_get_max_eig.restype = ctypes.c_float
    # extlib.fastmg_cheby_poly.argtypes = [ctypes.c_float, ctypes.c_float]
    extlib.fastmg_setup_smoothers.argtypes = [c_int]
    extlib.fastmg_update_A0.argtypes = [arr_float]

    extlib.fastFill_set_data.argtypes = [arr2d_int, c_int, arr_float, c_int, arr2d_float, c_float]
    extlib.fastFill_run.argtypes = [arr2d_float]
    extlib.fastFill_init.restype = c_int
    extlib.fastFill_fetch_A.argtypes = [arr_float, arr_int, arr_int]


    extlib.fastmg_new()

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


# def init_adj_edge(edges:np.ndarray,):
#     # 构建数据结构
#     vertex_to_edges = {}
#     for edge_index, (v1, v2) in enumerate(edges):
#         if v1 not in vertex_to_edges:
#             vertex_to_edges[v1] = set()
#         if v2 not in vertex_to_edges:
#             vertex_to_edges[v2] = set()
        
#         vertex_to_edges[v1].add(edge_index)
#         vertex_to_edges[v2].add(edge_index)

#     # 查找某条边相邻的边
#     def find_adjacent_edges(edge_index):
#         v1, v2 = edges[edge_index]
#         adjacent_edges = vertex_to_edges[v1] | vertex_to_edges[v2]  # 合并两个集合
#         adjacent_edges.remove(edge_index)  # 移除边本身
#         return list(adjacent_edges)

#     # 示例：查找第0条边的相邻边
#     print(find_adjacent_edges(0))
#     print(find_adjacent_edges(1))
#     print(find_adjacent_edges(2))
#     print(find_adjacent_edges(3))


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

    return all_adjacent_edges

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

ResidualLess = namedtuple('ResidualLess', ['dual', 'obj', 't']) 

def substep_xpbd(max_iter):
    semi_euler(old_pos, inv_mass, vel, pos)
    reset_lagrangian(lagrangian)

    calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
    fulldual0 = dual_residual.to_numpy()

    r = []
    for ite in range(max_iter):
        tic_iter = perf_counter()
        reset_dpos(dpos)
        solve_constraints_xpbd(dual_residual, inv_mass, edge, rest_len, lagrangian, dpos, pos)
        update_pos(inv_mass, dpos, pos)
        
        calc_r_xpbd = True
        if calc_r_xpbd:
            tic_calcr = perf_counter()
            t_iter = perf_counter()-tic_iter
            # calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
            dualr = np.linalg.norm(dual_residual.to_numpy()).astype(float)
            # compute_potential_energy()
            # compute_inertial_energy()
            # robj = (potential_energy[None]+inertial_energy[None])
            
            if ite==0:
                dualr0 = dualr

            if export_fullr:
                np.savez(out_dir+'/r/'+ f'fulldual_{frame}-{ite}', fulldual0)

            # r.append(ResidualLess(dual_r, robj, t_iter))
            if args.export_log:
                logging.info(f"{frame}-{ite}  dualr0:{dualr0:.2e} dual:{dualr:.2e}  t:{t_iter:.2e}s calcr:{perf_counter()-tic_calcr:.2e}s")

        if dualr < 0.1*dualr0 or dualr<1e-5:
            break

    if args.export_residual:
        tic = time.perf_counter()
        serialized_r = [r[i]._asdict() for i in range(len(r))]
        r_json = json.dumps(serialized_r)
        with open(out_dir+'/r/'+ f'{frame}.json', 'w') as file:
            file.write(r_json)
        print(f"export residual time: {time.perf_counter()-tic:.2f}s")

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


# DEPRECATED: build_levels_cuda is not needed any more because fastmg_RAP has done its job
# def build_levels_cuda(A, Ps=[]):
#     '''Give A and a list of prolongation matrices Ps, return a list of levels'''
#     lvl = len(Ps) + 1 # number of levels

#     levels = [MultiLevel() for i in range(lvl)]

#     levels[0].A = A

#     for i in range(lvl-1):
#         levels[i].P = Ps[i]
#     return levels



def setup_smoothers(A):
    global chebyshev_coeff
    if smoother_type == 'chebyshev':
        setup_chebyshev(A, lower_bound=1.0/30.0, upper_bound=1.1, degree=3)
    elif smoother_type == 'jacobi':
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


def new_amg_cg_solve(b, x0=None, tol=1e-5, maxiter=100):
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
    residuals = np.empty(shape=(maxiter+1,), dtype=np.float32)
    niter = extlib.fastmg_get_data(x, residuals)
    residuals = residuals[:niter+1]
    print(f"    niter", niter)
    print(f"    solve time: {time.perf_counter()-tic4:.3f}s")
    return (x),  residuals  


def diag_sweep(A,x,b,iterations=1):
    diag = A.diagonal()
    diag = np.where(diag==0, 1, diag)
    x[:] = b / diag

def presmoother(A,x,b):
    from pyamg.relaxation.relaxation import gauss_seidel, jacobi, sor, polynomial
    if smoother_type == 'gauss_seidel':
        gauss_seidel(A,x,b,iterations=1, sweep='symmetric')
    elif smoother_type == 'jacobi':
        jacobi(A,x,b,iterations=10, omega=jacobi_omega)
    elif smoother_type == 'sor_vanek':
        for _ in range(1):
            sor(A,x,b,omega=1.0,iterations=1,sweep='forward')
            sor(A,x,b,omega=1.85,iterations=1,sweep='backward')
    elif smoother_type == 'sor':
        sor(A,x,b,omega=1.33,sweep='symmetric',iterations=1)
    elif smoother_type == 'diag_sweep':
        diag_sweep(A,x,b,iterations=1)
    elif smoother_type == 'chebyshev':
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
    if use_primary_residual:
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



# has_run_initialize_fastFill = False
# def fill_A_by_spmm_dll(M_inv, ALPHA):
#     # -----------------initialize--------------
#     def initialize_fastFill(M_inv):
#         global extlib, has_run_initialize_fastFill, argtypes_of_csr
#         if has_run_initialize_fastFill:
#             return

#         extlib.fastA_new() # new fastA instance

#         M_inv = M_inv.tocsr() # transfer M to cusparse
#         extlib.fastA_set_M.argtypes = argtypes_of_csr
#         extlib.fastA_set_M(M_inv.data, M_inv.indices, M_inv.indptr, M_inv.shape[0], M_inv.shape[1], M_inv.nnz)
#         has_run_initialize_fastFill = True
    
#     initialize_fastFill(M_inv)
    
#     # -----------------fill G--------------
#     G_ii, G_jj, G_vv = np.zeros(NCONS*6, dtype=np.int32), np.zeros(NCONS*6, dtype=np.int32), np.zeros(NCONS*6, dtype=np.float32)
#     compute_C_and_gradC_kernel(pos, gradC, edge, constraints, rest_len)
#     fill_gradC_triplets_kernel(G_ii, G_jj, G_vv, gradC, edge)
#     G = scipy.sparse.csr_matrix((G_vv, (G_ii, G_jj)), shape=(NCONS, 3 * NV))

#     # transfer G to cusparse
#     extlib.fastA_set_G.argtypes = argtypes_of_csr
#     extlib.fastA_set_G(G.data, G.indices, G.indptr, G.shape[0], G.shape[1], G.nnz) 

#     # -----------------compute GMG by cusparse--------------
#     tic = time.perf_counter()
#     extlib.fastA_compute_GMG()
#     print(f"    GMG: {time.perf_counter() - tic:.4f}s")

#     # fetch GMG
#     from scipy.sparse import csr_matrix
#     mat = csr_matrix((G.shape[0],G.shape[0]), dtype=np.float32)
#     mat.data = np.empty(20* mat.shape[0], dtype=np.float32)
#     mat.indices = np.empty(20* mat.shape[0], dtype=np.int32)
#     extlib.fastA_fetch_A.argtypes = argtypes_of_csr[:3]
#     extlib.fastA_fetch_A(mat.data, mat.indices, mat.indptr)
#     mat = csr_matrix((mat.data, mat.indices, mat.indptr), shape=mat.shape)

#     # -----------------plus alpha--------------
#     A = mat + ALPHA
#     return A, G


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
        return False
    diff = A - B
    if diff.nnz == 0:
        return True
    maxdiff = np.abs(diff.data).max()
    print("maxdiff: ", maxdiff)
    if maxdiff > 1e-6:
        return False
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


# def fill_A_csr_cuda_and_fetch():
#     extlib.fastFill_run(pos.to_numpy())
#     extlib.fastFill_fetch_A(spmat_data, spmat_indices, spmat_indptr)
#     A = scipy.sparse.csr_matrix((spmat_data, spmat_indices, spmat_indptr), shape=(NE, NE))
#     return A

def fastFill_fetch():
    global spmat_data, spmat_indices, spmat_indptr
    extlib.fastFill_fetch_A(spmat_data, spmat_indices, spmat_indptr)
    A = scipy.sparse.csr_matrix((spmat_data, spmat_indices, spmat_indptr), shape=(NE, NE))
    return A


def fastFill_run():
    extlib.fastFill_run(pos.to_numpy())


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


def python_AMG(A,b):
    global Ps, num_levels

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
    x, r_Axb = old_amg_cg_solve(levels, b, x0=x0, maxiter=max_iter_Axb, tol=1e-6)
    toc = time.perf_counter()
    logging.info(f"    mgsolve time {toc-tic}")
    return  x, r_Axb


def calc_conv(r):
    return (r[-1]/r[0])**(1.0/(len(r)-1))

def report_multilevel_details(r_Axb, Ps, num_levels):

    logging.info(f"    num_levels:{num_levels}")
    num_points_level = []
    for i in range(len(Ps)):
        num_points_level.append(Ps[i].shape[0])
    num_points_level.append(Ps[-1].shape[1])
    for i in range(num_levels):
        logging.info(f"    num points of level {i}: {num_points_level[i]}")
    convfactor = calc_conv(r_Axb)
    logging.info(f"    convfactor: {convfactor:.2f}")


def should_setup():
    return ((frame%args.setup_interval==0) and (ite==0))


def calc_r_AMG(r,fulldual0, t_iter_start, r_Axb):
    t_iter = perf_counter()-t_iter_start
    tic_calcr = perf_counter()
    calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
    dual_r = np.linalg.norm(dual_residual.to_numpy()).astype(float)
    compute_potential_energy()
    compute_inertial_energy()
    robj = (potential_energy[None]+inertial_energy[None])
    r_Axb = r_Axb.tolist()
    # if use_primary_residual:
    #     primary_residual = calc_primary_residual(G, M_inv)
    #     primary_r = np.linalg.norm(primary_residual).astype(float)

    if export_fullr:
        fulldual_final = dual_residual.to_numpy()
        np.savez_compressed(out_dir+'/r/'+ f'fulldual_{frame}-{ite}', fulldual0, fulldual_final)

    logging.info(f"    convergence factor: {calc_conv(r_Axb):.2f}")
    logging.info(f"    Calc r time: {perf_counter()-tic_calcr:.4f}s")

    if args.export_log:
        logging.info(f"{frame}-{ite} rsys:{r_Axb[0]:.2e} {r_Axb[-1]:.2e} dual:{dual_r:.2e} object:{robj:.2e} iter:{len(r_Axb)} t:{t_iter:.3f}s")
    r.append(Residual([0.,0.], dual_r, robj, r_Axb, len(r_Axb), t_iter))



def do_export_r(r):
    if args.export_residual:
        serialized_r = [r[i]._asdict() for i in range(len(r))]
        r_json = json.dumps(serialized_r)
        with open(out_dir+'/r/'+ f'{frame}.json', 'w') as file:
            file.write(r_json)


Residual = namedtuple('residual', ['sys', 'dual', 'obj', 'r_Axb', 'niters','t'])


def substep_all_solver(max_iter=1):
    global pos, lagrangian, ite, t_export
    semi_euler(old_pos, inv_mass, vel, pos)
    reset_lagrangian(lagrangian)


    x = np.zeros(NCONS)
    r = []
    calc_dual_residual(dual_residual, edge, rest_len, lagrangian, pos)
    fulldual0 = dual_residual.to_numpy()
    for ite in range(max_iter):
        tic_assemble = perf_counter()
        t_iter_start = perf_counter()
        copy_field(pos_mid, pos)

        compute_C_and_gradC_kernel(pos, gradC, edge, constraints, rest_len)

        tic2 = perf_counter()
        if args.use_fastFill:
            fastFill_run()
        else:
            A = fill_A_csr_ti()
            # A,G = fill_A_by_spmm(M_inv, ALPHA)
        print(f"    fill_A time: {perf_counter()-tic2:.4f}s")

        update_constraints_kernel(pos, edge, rest_len, constraints)
        b = -constraints.to_numpy() - alpha_tilde_np * lagrangian.to_numpy()

        # #we calc inverse mass times gg(primary residual), because NCONS may contains infinity for fixed pin points. And gg always appears with inv_mass.
        # if use_primary_residual:
        #     Minv_gg =  (pos.to_numpy().flatten() - predict_pos.to_numpy().flatten()) - M_inv @ G.transpose() @ lagrangian.to_numpy()
        #     b += G @ Minv_gg
        logging.info(f"    Assemble matrix time: {perf_counter()-tic_assemble:.4f}s")

        if args.export_matrix:
            export_A_b(A,b,postfix=f"F{frame}-{ite}")

        if solver_type == "Direct":
            x = scipy.sparse.linalg.spsolve(A, b)
        if solver_type == "GS":
            gauss_seidel(A, x, b, iterations=max_iter_Axb, residuals=r_Axb)
        if solver_type == "AMG":
            if not args.use_cuda:
                x, r_Axb = python_AMG(A,b)
            else:
                global Ps, num_levels

                if should_setup():
                    tic = time.perf_counter()
                    if args.use_fastFill:
                        A = fastFill_fetch()
                    Ps = build_Ps(A)
                    num_levels = len(Ps)+1
                    logging.info(f"    build_Ps time:{time.perf_counter()-tic}")
                
                    extlib.fastmg_setup_nl(num_levels)

                    tic = time.perf_counter()
                    update_P(Ps)
                    print(f"    update_P time: {time.perf_counter()-tic:.2f}s")

                    tic = time.perf_counter()
                    cuda_set_A0(A)
                    extlib.fastmg_setup_smoothers(1) # 1 means chebyshev
                    logging.info(f"    setup smoothers time:{perf_counter()-tic}")

                # set A0
                tic = time.perf_counter()
                if args.use_fastFill:
                    extlib.fastmg_set_A0_from_fastFill()
                else:
                    cuda_update_A0(A)
                logging.info(f"    set_A0 time:{perf_counter()-tic:.3f}s")
                
                # compute RAP(R=P.T) and build levels in cuda-end
                tic3 = time.perf_counter()
                for lv in range(num_levels-1):
                    extlib.fastmg_RAP(lv) # build_levels is implicitly done 
                print(f"    RAP time: {time.perf_counter()-tic3:.3f}s")

                x0 = np.zeros_like(b)
                x, r_Axb = new_amg_cg_solve(b, x0=x0, maxiter=max_iter_Axb, tol=1e-5)

                if should_setup():
                    report_multilevel_details(r_Axb, Ps, num_levels)

        tic = time.perf_counter()
        transfer_back_to_pos_mfree(x)
        # if use_primary_residual:
        #     transfer_back_to_pos_matrix(x, M_inv, G, pos_mid, Minv_gg) #Chen2023 Primal XPBD
        print(f"    dlam to dpos time: {perf_counter()-tic:.4f}s")

        tic = perf_counter()
        calc_r_AMG(r, fulldual0,t_iter_start, r_Axb)
        do_export_r(r)
        t_export += perf_counter()-tic

        if r[-1].dual < 0.1*r[0].dual or r[-1].dual<1e-5:
            break
    update_vel(old_pos, inv_mass, vel, pos)


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
    print(f"removing {len(to_remove)} files")
    for file_path in to_remove:
        os.remove(file_path)
    print(f"clean {folder_path} done")
    os.chdir(pwd)

def create_another_outdir(out_dir):
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
    path.mkdir(parents=True, exist_ok=True)
    out_dir = str(path)
    print(f"\ncreate another outdir: {out_dir}\n")
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
    import datetime
    d = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"datetime:{d}",)
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
    arr = np.array([item + [-1]*(max_len - len(item)) if len(item) < max_len else item[:max_len] for item in d.values()])
    return arr, lengths


if args.auto_another_outdir:
    out_dir = create_another_outdir(out_dir)
    dont_clean_results = True


def init_set_P_R_manually():
    if solver_type=="AMG":
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


# SpMatData = namedtuple("SpMatData",['adj','nadj','adjabc','nnz','data','indices','indptr','ii','jj'])
# spmatdata = None
def init_direct_fill_A():
    # global spmatdata
    tic1 = perf_counter()
    print("Initializing adjacent edge and abc...")
    adjacent_edge = init_adj_edge(edges=edge.to_numpy())
    adjacent_edge,num_adjacent_edge = dict_to_ndarr(adjacent_edge)
    print(f"init_adjacent_edge time: {perf_counter()-tic1:.3f}s")

    adjacent_edge_abc = np.empty((NE, 20*3), dtype=np.int32)
    adjacent_edge_abc.fill(-1)
    init_adjacent_edge_abc_kernel(NE,edge,adjacent_edge,num_adjacent_edge,adjacent_edge_abc)
    # required by fill_A_csr_ti

    tic2 = perf_counter()
    #calculate number of nonzeros by counting number of adjacent edges
    num_nonz = calc_num_nonz(num_adjacent_edge) 
    nnz_each_row = calc_nnz_each_row(num_adjacent_edge)

    # init csr pattern. In the future we will replace all ijv pattern with csr
    data, indices, indptr = init_A_CSR_pattern(num_adjacent_edge, adjacent_edge)
    ii, jj = csr_index_to_coo_index(indptr, indices)

    # spMatA = SpMat(num_nonz, NE)
    # spMatA._init_pattern()
    # fill_A_diag_kernel(spMatA.diags)

    print(f"init A CSR pattern time: {perf_counter()-tic2:.3f}s")
    print(f"init_direct_fill_A time: {perf_counter()-tic1:.3f}s")
    # spmatdata = SpMatData(adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, data, indices, indptr, ii, jj)
    print("caching init_direct_fill_A...")
    tic = perf_counter() # savez_compressed will save 10x space(1.4G->140MB), but much slower(33s)
    np.savez(f'cache_initFill_N{N}.npz', adjacent_edge=adjacent_edge, num_adjacent_edge=num_adjacent_edge, adjacent_edge_abc=adjacent_edge_abc, num_nonz=num_nonz, spmat_data=data, spmat_indices=indices, spmat_indptr=indptr, spmat_ii=ii, spmat_jj=jj)
    print("time of caching:", perf_counter()-tic)
    return adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, data, indices, indptr, ii, jj


def init_direct_fill_A_cuda():
    extlib.fastFill_new()

    extlib.fastFill_set_data(edge.to_numpy(), NE, inv_mass.to_numpy(), NV, pos.to_numpy(), alpha)
    nonz = extlib.fastFill_init()

    global spmat_data, spmat_indices, spmat_indptr
    spmat_indptr = np.empty(NE+1, dtype=np.int32)
    spmat_indices = np.empty(nonz, dtype=np.int32)
    spmat_data = np.empty(nonz, dtype=np.float32)



def cache_and_init_direct_fill_A():
    global adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, spmat_data, spmat_indices, spmat_indptr, spmat_ii, spmat_jj
    if  os.path.exists(f'cache_initFill_N{N}.npz') and args.use_cache:
        npzfile= np.load(f'cache_initFill_N{N}.npz')
        (adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, spmat_data, spmat_indices, spmat_indptr, spmat_ii, spmat_jj) = (npzfile[key] for key in ['adjacent_edge', 'num_adjacent_edge', 'adjacent_edge_abc', 'num_nonz', 'spmat_data', 'spmat_indices', 'spmat_indptr', 'spmat_ii', 'spmat_jj'])
        num_nonz = int(num_nonz) # npz save int as np.array, it will cause bug in taichi kernel
        print(f"load cache_initFill_N{N}.npz")
    else:
        adjacent_edge, num_adjacent_edge, adjacent_edge_abc, num_nonz, spmat_data, spmat_indices, spmat_indptr, spmat_ii, spmat_jj = init_direct_fill_A()



def ending(timer_loop, start_date, initial_frame, t_export_total):
    t_all = time.perf_counter() - timer_loop
    end_date = datetime.datetime.now()
    end_frame = frame
    s = f"\n-------\nTime: {(time.perf_counter() - timer_loop):.2f}s = {(time.perf_counter() - timer_loop)/60:.2f}min. \nFrame {initial_frame}-{end_frame}({end_frame-initial_frame} frames). \nAvg: {t_all/(end_frame-initial_frame):.2f}s/frame. \nStart\t{start_date},\nEnd\t{end_date}.\nTime of exporting: {t_export_total:.3f}s"
    if args.export_log:
        logging.info(s)


misc_dir_path = prj_path + "/data/misc/"
mkdir_if_not_exist(out_dir)
mkdir_if_not_exist(out_dir + "/r/")
mkdir_if_not_exist(out_dir + "/A/")
mkdir_if_not_exist(out_dir + "/state/")
mkdir_if_not_exist(out_dir + "/mesh/")
if not args.restart and not dont_clean_results:
    clean_result_dir(out_dir)
    clean_result_dir(out_dir + "/r/")
    clean_result_dir(out_dir + "/A/")
    clean_result_dir(out_dir + "/state/")
    clean_result_dir(out_dir + "/mesh/")


logging.basicConfig(level=logging.INFO, format="%(message)s",filename=out_dir + f'/latest.log',filemode='a')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


print_all_globals(global_vars)

# ---------------------------------------------------------------------------- #
#                                initialization                                #
# ---------------------------------------------------------------------------- #
timer_all = time.perf_counter()
start_wall_time = datetime.datetime.now()


def init():
    global frame
    print("\nInitializing...")
    print("Initializing pos..")
    init_pos(inv_mass,pos)
    init_tri(tri)
    init_edge(edge, rest_len, pos)
    write_mesh(out_dir + f"/mesh/{frame:04d}", pos.to_numpy(), tri.to_numpy())
    print("Initializing pos and edge done")
    if args.setup_num == 1:
        init_scale()

    tic = time.perf_counter()
    if args.use_cuda and args.use_fastFill:
        init_direct_fill_A_cuda()
    else:
        cache_and_init_direct_fill_A()
    print(f"init_direct_fill_A time: {time.perf_counter()-tic:.3f}s")

    if args.restart:
        if args.restart_from_last_frame :
            args.restart_frame =  find_last_frame(out_dir)
        if args.restart_frame == 0:
            print("No restart file found.")
        else:
            load_state(args.restart_dir + f"{args.restart_frame:04d}.npz")
            frame = args.restart_frame
            logging.info(f"restart from last frame: {args.restart_frame}")

    print(f"Initialization done. Cost time:  {time.perf_counter() - timer_all:.3f}s") 


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

viewer = Viewer()

def run():
    global frame, paused, ite
    timer_loop = time.perf_counter()
    initial_frame = frame
    step_pbar = tqdm.tqdm(total=end_frame, initial=frame)
    t_export_total = 0.0
    try:
        while True:
            tic_frame = time.perf_counter()
            t_export = 0.0

            if use_viewer:
                for e in viewer.window.get_events(ti.ui.PRESS):
                    if e.key in [ti.ui.ESCAPE]:
                        exit()
                    if e.key == ti.ui.SPACE:
                        paused = not paused
                        print("paused:",paused)
            if not paused:
                if solver_type == "XPBD":
                    substep_xpbd(max_iter)
                else:
                    substep_all_solver(max_iter)
                frame += 1
                
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
            if frame == end_frame:
                print("Normallly end.")
                ending(timer_loop, start_wall_time, initial_frame, t_export_total)
                exit()
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