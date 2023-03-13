import scipy.io as sio
P = sio.mmread("models/bunny1000_2000/P.mtx")
R = sio.mmread("models/bunny1000_2000/P.mtx")
PT = P.transpose()
# if (PT - R).nnz == 0:
#     print("P and R are the same")

R_dense = R.todense()

pass