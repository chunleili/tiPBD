import numpy as np
import taichi as ti
from time import perf_counter
import os
import logging

def dict_to_ndarr(d:dict)->np.ndarray:
    lengths = np.array([len(v) for v in d.values()])

    max_len = max(len(item) for item in d.values())
    # 使用填充或截断的方式转换为NumPy数组
    arr = np.array([list(item) + [-1]*(max_len - len(item)) if len(item) < max_len else list(item)[:max_len] for item in d.values()])
    return arr, lengths

class SpMat:
    def get_from_outside(self, data, indices, indptr, ii, jj):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.ii = ii
        self.jj = jj
        assert  np.all(jj==indices)

    def create(self, nnz,num_rows):
        self.indptr = np.zeros(num_rows+1, dtype=np.int32)
        self.data = np.zeros(nnz, dtype=np.float32)
        self.indices = np.zeros(nnz, dtype=np.int32)
        self.ii = np.zeros(nnz, dtype=np.int32)
        self.jj = self.indices

    @property
    def nnz(self):
        return len(self.data)
    
    def get_nrows(self):
        return len(self.indptr)-1


@ti.data_oriented
class FillACloth():
    def __init__(self, pos, inv_mass, edge, alpha, use_cache, use_cuda, extlib=None) -> None:
        self.pos = pos
        self.inv_mass = inv_mass
        self.edge = edge
        self.alpha = alpha
        self.NV = pos.shape[0]
        self.NE = edge.shape[0]
        self.extlib =  extlib
        self.use_cuda = use_cuda
        self.use_cache = use_cache

        self.cache_name = f"cache_cloth_NV{self.NV}_NE{self.NE}.npz"

        if use_cuda:
            self.use_cpp_initFill = True
        else:
            self.use_cpp_initFill = False


        self.spmat = SpMat()

    def init(self):
        if self.use_cuda:
            self.load_cache_initFill_to_cuda()
        else:
            self.cache_and_initFill()

    def load(self):
        self.cache_and_initFill()

    def initFill_python(self):
        edge = self.edge
        NE = edge.shape[0]

        tic1 = perf_counter()
        print("Initializing adjacent edge and abc...")
        self.adjacent_edge, v2e_dict = self.init_adj_edge(edges=edge.to_numpy())
        self.adjacent_edge,self.num_adjacent_edge = dict_to_ndarr(self.adjacent_edge)
        MAX_ADJ = self.adjacent_edge.shape[1]
        self.MAX_ADJ =MAX_ADJ
        logging.info(f"MAX_ADJ:{MAX_ADJ}")
        self.v2e, self.num_v2e = dict_to_ndarr(v2e_dict)
        self.adjacent_edge_abc = np.zeros((NE, MAX_ADJ*3), dtype=np.int32)
        self.init_adjacent_edge_abc_kernel(NE,edge,self.adjacent_edge,self.num_adjacent_edge,self.adjacent_edge_abc)
        self.num_nonz = self.calc_num_nonz(self.num_adjacent_edge).astype(np.int32)
        data, indices, indptr = self.init_A_CSR_pattern(self.num_adjacent_edge, self.adjacent_edge, NE)
        ii, jj = self.csr_index_to_coo_index(indptr, indices)
        self.spmat.get_from_outside(data, indices, indptr, ii, jj)
        print(f"initFill time: {perf_counter()-tic1:.3f}s")

    def get_alldata(self):
        return  (self.adjacent_edge,
            self.num_adjacent_edge,
            self.adjacent_edge_abc,
            self.num_nonz,
            self.spmat.indices,
            self.spmat.indptr,
            self.spmat.ii,
            self.spmat.jj,
            self.v2e,
            self.num_v2e,)

    def alldata_names(self):
        return ["adjacent_edge",
            "num_adjacent_edge",
            "adjacent_edge_abc",
            "num_nonz",
            "spmat_indices",
            "spmat_indptr",
            "spmat_ii",
            "spmat_jj",
            "v2e",
            "num_v2e",
            ]

    def alldata_dict(self):
        return dict(zip(self.alldata_names(), self.get_alldata()))

    def initFill_cpp(self):
        edge = self.edge
        NE = edge.shape[0]
        extlib = self.extlib

        NV = self.NV
        MAX_ADJ = 20
        MAX_V2E = MAX_ADJ
        adjacent_edge_abc = np.empty((NE, 20*3), dtype=np.int32)
        adjacent_edge_abc.fill(-1)
        self.adjacent_edge = np.zeros((NE, MAX_ADJ), dtype=np.int32)
        self.num_adjacent_edge = np.zeros(NE, dtype=np.int32)
        self.adjacent_edge_abc = np.zeros((NE, MAX_ADJ*3), dtype=np.int32)
        self.v2e = np.zeros((NV, MAX_V2E), dtype=np.int32)
        self.num_v2e = np.zeros(NV, dtype=np.int32)

        tic1 = perf_counter()
        print("Initializing adjacent edge and abc...")
        extlib.initFillCloth_set(edge.to_numpy(), NE)
        extlib.initFillCloth_run()
        num_nonz = extlib.initFillCloth_get_nnz()
        self.num_nonz = num_nonz
        self.spmat.create(num_nonz, NE)
        extlib.initFillCloth_get(*self.get_alldata())
        print(f"initFill time: {perf_counter()-tic1:.3f}s")

    def push_data(self):
        return self.spmat

    def load_cache(self):
        npzfile = np.load(f"{self.cache_name}")
        for key in self.alldata_names():
            setattr(self, key, npzfile[key])
        self.spmat.jj = self.spmat.indices
        self.num_nonz = self.calc_num_nonz(self.num_adjacent_edge)
        print(f"load {self.cache_name}")

    def save_cache(self):
        print("caching init fill...")
        tic = perf_counter() 
        alldata = self.get_alldata()
        np.savez(f"{self.cache_name}", **dict(zip(self.alldata_names(), alldata)))
        print("time of caching:", perf_counter() - tic)

    def cache_and_initFill(self):
        if  self.use_cache and os.path.exists(f'{self.cache_name}') :
            self.load_cache()
        else:
            if self.use_cuda and self.use_cpp_initFill:
                self.initFill_cpp()
            else:
                self.initFill_python()
            self.save_cache()

    @staticmethod
    def calc_num_nonz(num_adjacent_edge):
        num_nonz = np.sum(num_adjacent_edge)+num_adjacent_edge.shape[0]
        return num_nonz

    @staticmethod
    def calc_nnz_each_row(num_adjacent_edge):
        nnz_each_row = num_adjacent_edge[:] + 1
        return nnz_each_row

    @staticmethod
    def init_A_CSR_pattern(num_adjacent_edge, adjacent_edge, NE):
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

    @staticmethod
    def csr_index_to_coo_index(indptr, indices):
        ii, jj = np.zeros_like(indices), np.zeros_like(indices)
        for i in range(indptr.shape[0]-1):
            ii[indptr[i]:indptr[i+1]]=i
            jj[indptr[i]:indptr[i+1]]=indices[indptr[i]:indptr[i+1]]
        return ii, jj

    def load_cache_initFill_to_cuda(self):
        self.cache_and_initFill()
        self.extlib.fastFillCloth_set_data(
            self.edge.to_numpy(),
            self.NE,
            self.inv_mass.to_numpy(),
            self.NV,
            self.pos.to_numpy(),
            self.alpha,
        )
        self.extlib.fastFillCloth_init_from_python_cache(self.adjacent_edge,
                                            self.num_adjacent_edge,
                                            self.adjacent_edge_abc,
                                            self.num_nonz,
                                            self.spmat.data,
                                            self.spmat.indices,
                                            self.spmat.indptr,
                                            self.spmat.ii,
                                            self.spmat.jj,
                                            self.NE,
                                           self.NV)
        # self.extlib.fastFillCloth_init_from_python_cache(*self.get_alldata()[:-2], self.NE, self.NV)


    def compare_find_shared_v_order(v,e1,e2,edge):
        # which is shared v in e1? 0 or 1
        order_in_e1 = 0 if edge[e1][0] == v else 1
        order_in_e2 = 0 if edge[e2][0] == v else 1
        return order_in_e1, order_in_e2

    @staticmethod
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

    @staticmethod
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
