import os
import ctypes
import numpy.ctypeslib as ctl
import numpy as np
import subprocess


def init_extlib(args, sim=""):
    prj_path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if args.debug:
        os.chdir(prj_path+'/cpp/mgcg_cuda')
        retcode = subprocess.call(["cmake", "--build", "build", "--config", "Debug", "--parallel", "8"])
        if retcode != 0:
            raise Exception("Failed to build the project")
        os.chdir(prj_path)

    os.add_dll_directory(args.cuda_dir)
    extlib = ctl.load_library("fastmg.dll", prj_path+'/cpp/mgcg_cuda/lib')

    arr_int = ctl.ndpointer(dtype=np.int32, ndim=1, flags='aligned, c_contiguous')
    arr_float = ctl.ndpointer(dtype=np.float32, ndim=1, flags='aligned, c_contiguous')
    arr2d_float = ctl.ndpointer(dtype=np.float32, ndim=2, flags='aligned, c_contiguous')
    arr2d_int = ctl.ndpointer(dtype=np.int32, ndim=2, flags='aligned, c_contiguous')
    arr3d_float = ctl.ndpointer(dtype=np.float32, ndim=3, flags='aligned, c_contiguous')
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
    extlib.fastmg_set_smoother_niter.argtypes = [ctypes.c_size_t]

    extlib.fastmg_get_nnz.argtypes = [ctypes.c_int]
    extlib.fastmg_get_nnz.restype = ctypes.c_int
    extlib.fastmg_get_matsize.argtypes = [ctypes.c_int]
    extlib.fastmg_get_matsize.restype = ctypes.c_int
    extlib.fastmg_fetch_A.argtypes = [ctypes.c_int, arr_float, arr_int, arr_int]
    extlib.fastmg_fetch_A_data.argtypes = [arr_float]
    extlib.fastmg_use_radical_omega.argtypes = [ctypes.c_int]


    extlib.fastmg_new()
    if args.scale_RAP:
        extlib.fastmg_scale_RAP.argtypes = [c_float, c_int]

    extlib.fastmg_use_radical_omega(0)

    if sim=="cloth":
        extlib.fastFillCloth_set_data.argtypes = [arr2d_int, c_int, arr_float, c_int, arr2d_float, c_float]
        extlib.fastFillCloth_run.argtypes = [arr2d_float]
        extlib.fastFillCloth_fetch_A_data.argtypes = [arr_float]
        extlib.fastFillCloth_init_from_python_cache.argtypes = [arr2d_int, arr_int, arr2d_int, c_int, arr_float, arr_int, arr_int, arr_int, arr_int, c_int, c_int]
        extlib.initFillCloth_set.argtypes = [arr2d_int, c_int]
        extlib.initFillCloth_get.argtypes = [arr2d_int, arr_int, arr2d_int, c_int] + [arr_int]*4 + [arr2d_int, arr_int]
        extlib.initFillCloth_new()
        extlib.fastFillCloth_new()

    elif sim=="soft":
        extlib.fastFillSoft_set_data.argtypes = [arr2d_int, c_int, arr_float, c_int, arr2d_float, arr_float]
        extlib.fastFillSoft_fetch_A_data.argtypes = [arr_float]
        extlib.fastFillSoft_run.argtypes = [arr2d_float, arr3d_float]
        extlib.fastFillSoft_new()
    
    return extlib





