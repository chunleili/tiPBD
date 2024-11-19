import ctypes
import numpy as np
from time import perf_counter
from ctypes import c_int
from pathlib import Path
import os
import numpy.ctypeslib as ctl

# ---------------------------------------------------------------------------- #
#                             start graph coloring                             #
# ---------------------------------------------------------------------------- #
# We use v2 Now

# version 2, use pyamg.
# Input: CSR matrix(symmetric)
# This is called in AMG_setup_phase()
def graph_coloring_v2(fetch_A_from_cuda, num_levels, extlib=None, model_path=None):
    # For caching the coloring result
    has_colored_L = [False]*num_levels
    if model_path is None:
        model_path = os.getcwd() # just cache in current dir
    dir = str(Path(model_path).parent) #cache in the model dir
    for lv in range(num_levels):
        path = dir+f'/coloring_L{lv}.txt'
        has_colored_L[lv] =  os.path.exists(path)
    has_colored = all(has_colored_L)
    if not has_colored:
        has_colored = True
    else:
        return

    # do coloring
    from pyamg.graph import vertex_coloring
    tic = perf_counter()
    for i in range(num_levels):
        print(f"level {i}")
        Ai = fetch_A_from_cuda(i)
        colors = vertex_coloring(Ai)
        ncolor = np.max(colors)+1
        print(f"ncolor: {ncolor}")
        print("colors:",colors)
        np.savetxt(dir + f"/color_L{i}.txt", colors, fmt="%d")
        if extlib is not None:
            graph_coloring_to_cuda(ncolor, colors, i, extlib)
    print(f"graph_coloring_v2 time: {perf_counter()-tic:.3f}s")
    return ncolor, colors


def graph_coloring_to_cuda(ncolor, colors, lv, extlib):
    colors = np.ascontiguousarray(colors)
    arr_int = ctl.ndpointer(dtype=np.int32, ndim=1, flags='aligned, c_contiguous')
    extlib.fastmg_set_colors.argtypes = [arr_int, c_int, c_int, c_int]
    extlib.fastmg_set_colors(colors, colors.shape[0], ncolor, lv)





# ---------------------------------------------------------------------------- #
#                              BELLOWING NOT USED                              #
# ---------------------------------------------------------------------------- #

# version 1, hand made. It is slow. By Wang Ruiqi.
# Input: .ele file
def graph_coloring_v1():
    extlib.graph_coloring.argtypes = [ctypes.c_char_p, arr_int ]
    extlib.restype = c_int
    colors = np.zeros(ist.NT, dtype=np.int32)
    abs_path = os.path.abspath(args.model_path)
    abs_path = abs_path.replace(".node", ".ele")
    model = abs_path.encode('ascii')
    tic = perf_counter()
    ncolor = extlib.graph_coloring(model, colors)
    print(f"ncolor: {ncolor}")
    print("colors of tets:",colors)
    print(f"graph_coloring_v1 time: {perf_counter()-tic:.3f}s")
    return ncolor, colors





# version 3, use newtworkx.
# Input: CSR matrix(symmetric)
# This is called in AMG_setup_phase()
def graph_coloring_v3(A):
    import networkx as nx
    tic = perf_counter()
    net = nx.from_scipy_sparse_array(A)
    colors = nx.coloring.greedy_color(net)
    # change colors from dict to numpy array
    colors = np.array([colors[i] for i in range(len(colors))])
    ncolor = np.max(colors)+1
    print(f"ncolor: {ncolor}")
    print("colors:",colors)
    print(f"graph_coloring_v3 time: {perf_counter()-tic:.3f}s")
    return ncolor, colors


# read the color.txt
# Input: color.txt file path
def graph_coloring_read():
    model_dir = Path(args.model_path).parent
    path = model_dir / "color.txt"
    tic = perf_counter()

    require_process = True
    if require_process: #ECL_GC, # color.txt is nx3, left is node index, right is color
        colors_raw = np.loadtxt(path, dtype=np.int32, skiprows=1)
        # colors = colors_raw[:,0:2] # get first and third column
        # sort by node index
        sorted_indices = np.argsort(colors_raw[:, 0])
        sorted_colors = colors_raw[sorted_indices]
        colors = sorted_colors[:, 2]
    else: # ruiqi, no need to process
        colors = np.loadtxt(path, dtype=np.int32)

    ncolor = np.max(colors)+1
    print(f"ncolor: {ncolor}")
    print("colors:",colors)
    print(f"graph_coloring_read time: {perf_counter()-tic:.3f}s")


    graph_coloring_to_cuda(ncolor, colors,0)

    return ncolor, colors



