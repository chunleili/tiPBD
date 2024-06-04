#%%
from scipy.sparse import csr_matrix
from pyamg.amg_core import apply_distance_filter
from scipy import array
# Graph in CSR where entries in row i represent distances from dof i
indptr = array([0,3,6,9])
indices = array([0,1,2,0,1,2,0,1,2])
data = array([1.,2.,3.,4.,1.,2.,3.,9.,1.])
S = csr_matrix( (data,indices,indptr), shape=(3,3) )
# print "Matrix before Applying Filter\n" + str(S.todense())
# apply_distance_filter(3, 1.9, S.indptr, S.indices, S.data)
# print "Matrix after Applying Filter\n" + str(S.todense())