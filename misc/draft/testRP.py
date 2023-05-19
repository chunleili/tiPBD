import scipy.io as sio

P = sio.mmread("model/bunny1k2k/P.mtx")
R = sio.mmread("model/bunny1k2k/P.mtx")
PT = P.transpose()
# if (PT - R).nnz == 0:
#     print("P and R are the same")

R_dense = R.todense()

pass
