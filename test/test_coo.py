from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import numpy as np

n = 8
m = 3
lam = np.random.rand(m,1)

gradC_vec = np.random.rand(m,4,3)
print("gradC_vec")
print(gradC_vec)

#fill gradC
row = np.zeros(12*m, dtype=np.int32)
col = np.zeros(12*m, dtype=np.int32)
val = np.zeros(12*m, dtype=np.float32)
for j in range(m):
    # ia,ib,ic,id = np.random.randint(low=0, high=n,size=4)
    ia,ib,ic,id = 0,2,3,7
    val[12*j+0*3 : 12*j+0*3+3] = gradC_vec[j,0]
    val[12*j+1*3 : 12*j+1*3+3] = gradC_vec[j,1]
    val[12*j+2*3 : 12*j+2*3+3] = gradC_vec[j,2]
    val[12*j+3*3 : 12*j+3*3+3] = gradC_vec[j,3]
    row[12*j     : 12*j+12] = j 
    col[12*j+3*0 : 12*j+3*0+3] = 3*ia, 3*ia+1, 3*ia+2
    col[12*j+3*1 : 12*j+3*1+3] = 3*ib, 3*ib+1, 3*ib+2
    col[12*j+3*2 : 12*j+3*2+3] = 3*ic, 3*ic+1, 3*ic+2
    col[12*j+3*3 : 12*j+3*3+3] = 3*id, 3*id+1, 3*id+2

gradC_coo = coo_matrix((val, (row, col)), shape=(m, n*3))
print("gradC_coo\n",gradC_coo)
print("gradC_coo.todense()\n",gradC_coo.todense())