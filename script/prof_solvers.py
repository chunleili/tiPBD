"""测试profile各种solver的性能"""
import numpy as np
import os, sys
from utils.solvers import *
from utils.load_A_b import load_A_b

sys.path.append(os.getcwd())

save_fig = True
show_fig = False
show_time_plot = True
only_postprecess = False
run_concate_png = False

def test_amg(A, b, postfix=""):
    x0 = np.zeros_like(b)
    allres = []
    ml17=UA_CG(A, b,x0, allres)

if __name__ == "__main__":
    # frames = [1,6,11,16,21,26]
    frames = [10,20,30,40,50,60]
    ite = 0
    for frame in frames:
        postfix = f"F{frame}-{ite}"
        A,b = load_A_b(postfix=postfix)
        for _ in range(1):
            test_amg(A,b,postfix=postfix)

