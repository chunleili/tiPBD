import meshio
from scipy.io import mmwrite, mmread
from pathlib import Path
import numpy as np


readpath = Path("D:/Dev/tiPBD/result/scale64/obj/0001.obj")
outpath = Path("D:/Dev/tiPBD/result/scale64/obj/0001.mtx")
def mesh_to_mtx(readpath, outpath):
    mesh = meshio.read(readpath)
    p = mesh.points
    p = p.ravel(order='F')
    p = p.reshape((p.size, 1))
    mmwrite(outpath, p)


def txt_to_mtx(readpath, outpath):
    b = np.loadtxt(readpath, dtype=np.float64)
    b = b.reshape((b.size, 1))
    mmwrite(outpath, b)


readpath = Path("D:/Dev/tiPBD/result/scale64/A/b_F1-0.txt")
outpath = Path("D:/Dev/tiPBD/result/scale64/A/b_F1-0.mtx")
txt_to_mtx(readpath, outpath)