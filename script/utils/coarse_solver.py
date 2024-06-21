import numpy as np
from scipy.linalg import pinv
from time import perf_counter

# class CoarseSolver:
#     def __init__(self, method='pinv'):
#         self.method = method

#     # 实现仅第一次进入coarse_solver时计算一次P
#     def solve(self, A, b):
#         tic = perf_counter()
#         if not hasattr(self, 'P'):
#             self.P = pinv(A.toarray())
#         res = np.dot(self.P, b)
#         toc = perf_counter()
#         print('pyamg coarse_solver Time:', toc-tic)
#         return res


# 实现仅第一次进入coarse_solver时计算一次P
# https://stackoverflow.com/a/279597/19253199
def coarse_solver(A, b):
    if not hasattr(coarse_solver, "P"):
        coarse_solver.P = pinv(A.toarray())
    res = np.dot(coarse_solver.P, b)
    return res