import scipy
import numpy as np

# judge if A is positive definite
# https://stackoverflow.com/a/44287862/19253199
def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            print("A is positive definite")
            return True
        except np.linalg.LinAlgError:
            print("A is not positive definite")
            return False
    else:
        print("A is not positive definite")
        return False
    
if __name__ == "__main__":
    A = scipy.io.mmread("./result/test/A2.mtx")
    A=A.todense()
    is_pos_def(A)
    # chol_A = np.linalg.cholesky(A)
    # print(chol_A)