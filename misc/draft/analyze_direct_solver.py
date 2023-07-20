import numpy as np
import scipy
import time

start = time.time()
A = scipy.io.mmread("result/A_0.mtx")
print(A)

b = np.loadtxt("result/b_0.txt")
print(b)
A = A.tocsr()
print("time before solve: ", time.time() - start)
x = scipy.sparse.linalg.spsolve(A, b)
print("time after solve: ", time.time() - start)
