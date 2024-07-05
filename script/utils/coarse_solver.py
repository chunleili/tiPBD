import numpy as np
from scipy.linalg import pinv
from time import perf_counter


# # pinv实现仅第一次进入coarse_solver时计算一次P
# # https://stackoverflow.com/a/279597/19253199
# def coarse_solver(A, b):
#     if not hasattr(coarse_solver, "P"):
#         coarse_solver.P = pinv(A.toarray())
#     res = np.dot(coarse_solver.P, b)
#     return res


def coarse_solver(A, b):
    res = np.linalg.solve(A.toarray(), b)
    return res