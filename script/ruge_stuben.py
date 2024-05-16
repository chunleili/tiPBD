"""Build P based on Ruge-Stueben algorithm"""
import numpy as np
import scipy
from scipy.io import mmread, mmwrite
import scipy.sparse as sparse
import os, sys
from time import perf_counter
from matplotlib import pyplot as plt
import pyamg
from pyamg.gallery import poisson
from collections import namedtuple


def main(A):
    # A = poisson((100,)).tocsr()
    levels = ruge_stuben_solver_my(A, max_levels=2, max_coarse=2, keep=True)
    P = levels[0].P
    R = levels[0].R
    Ac = levels[1].A
    C = levels[0].C
    splitting = levels[0].splitting
    # print("A\n",A.toarray())
    print("P\n",P.toarray())
    # print("R\n",R.toarray())
    # print("Ac\n",Ac.toarray())
    print("C\n",C.toarray())
    print("splitting\n",splitting.astype(int))
    return P, splitting


def test_pyamg(A):
    # A = poisson((100,)).tocsr()
    ml = pyamg.ruge_stuben_solver(A, max_levels=2, max_coarse=2,keep=True)
    # ml = pyamg.ruge_stuben_solver(A, max_levels=2, max_coarse=2, keep=True)
    levels = ml.levels
    P = levels[0].P
    R = levels[0].R
    Ac = levels[1].A
    C = levels[0].C
    splitting = levels[0].splitting
    # print("A\n",A.toarray())
    print("P\n",P.toarray())
    # print("R\n",R.toarray())
    # print("Ac\n",Ac.toarray())
    # print("C\n",C.toarray())
    print("splitting\n",splitting.astype(int))
    return P, splitting


def test_classical_strength_of_connection():
    from pyamg.classical.classical import classical_strength_of_connection
    A = poisson((4,)).tocsr()
    print(A.toarray())
    S = classical_strength_of_connection(A, 0.0)
    print(S.toarray())



class Level:
    def __init__(self):
        self.A = None
        self.C = None
        self.P = None
        self.R = None


def ruge_stuben_solver_my(A,
                       strength=('classical', {'theta': 0.25}),
                       CF=('RS', {'second_pass': False}),
                       interpolation='classical',
                       max_levels=30, max_coarse=10, keep=False, **kwargs):
    # A is csr. A is square matrix
    levels = [Level()]
    levels[-1].A = A
    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        bottom = _extend_hierarchy(levels, strength, CF, interpolation, keep)
        if len(levels) >= max_levels:
            print("max_levels reached")
        if levels[-1].A.shape[0] <= max_coarse:
            print("max_coarse reached")
        if bottom:
            print("bottom reached")
            break
    return levels



def _extend_hierarchy(levels, strength, CF, interpolation, keep):
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}
    
    A = levels[-1].A

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j.
    fn, kwargs = unpack_arg(strength)
    if fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)

    # Generate the C/F splitting
    fn, kwargs = unpack_arg(CF)
    if fn == 'RS':
        splitting = RS(C, **kwargs)

    # Make sure all points were not declared as C- or F-points
    # Return early, do not add another coarse level
    num_fpts = np.sum(splitting)
    if (num_fpts == len(splitting)) or (num_fpts == 0):
        print("all points were C-points or all points are F-points, early return")
        return True

    # Generate the interpolation matrix that maps from the coarse-grid to the
    # fine-grid
    fn, kwargs = unpack_arg(interpolation)
    if fn == 'classical':
        P = classical_interpolation(A, C, splitting, **kwargs)

    # Generate the restriction matrix that maps from the fine-grid to the
    # coarse-grid
    R = P.T.tocsr()

    # Store relevant information for this level
    if keep:
        levels[-1].C = C                           # strength of connection matrix

    levels[-1].splitting = splitting.astype(bool)  # C/F splitting
    levels[-1].P = P                               # prolongation operator
    levels[-1].R = R                               # restriction operator

    # Form next level through Galerkin product
    levels.append(Level)
    A = R * A * P
    levels[-1].A = A
    return False


def classical_strength_of_connection(A, theta=0.1, block=True, norm='abs'):
    if (theta < 0 or theta > 1):
        raise ValueError('expected theta in [0,1]')

    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)
    
    data = A.data
    N = A.shape[0]

    Sp = np.empty_like(A.indptr)
    Sj = np.empty_like(A.indices)
    Sx = np.empty_like(data)

    amg_core_classical_strength_of_connection_abs(
            N, theta, A.indptr, A.indices, data, Sp, Sj, Sx)

    S = sparse.csr_matrix((Sx, Sj, Sp), shape=[N, N])

    # Take magnitude and scale by largest entry
    S.data = np.abs(S.data)
    S = scale_rows_by_largest_entry(S)
    S.eliminate_zeros()
    return S


def scale_rows_by_largest_entry(S):
    S = S.tocsr()
    for i in range(S.shape[0]):
        row_start = S.indptr[i]
        row_end = S.indptr[i + 1]
        if row_start == row_end:
            continue # empty row
        max_entry = np.max(np.abs(S.data[row_start:row_end]))
        if max_entry > 0:
            S.data[row_start:row_end] /= max_entry
    return S



def RS(S, second_pass=False):
    S = remove_diagonal(S)

    T = S.T.tocsr()  # transpose S for efficient column access
    splitting = np.empty(S.shape[0], dtype='intc')
    influence = np.zeros((S.shape[0],), dtype='intc')

    amg_core_rs_cf_splitting(S.shape[0],
                             S.indptr, S.indices,
                             T.indptr, T.indices,
                             influence,
                             splitting)
    return splitting

def remove_diagonal(S):
    assert S.shape[0] == S.shape[1]
    S = S.tocoo()
    mask = S.row != S.col
    S.row = S.row[mask]
    S.col = S.col[mask]
    S.data = S.data[mask]
    return S.tocsr()



def classical_interpolation(A, C, splitting, theta=None, norm='min', modified=True):
    """Create prolongator using distance-1 classical interpolation.

    Parameters
    ----------
    A : csr_matrix
        NxN matrix in CSR format
    C : csr_matrix
        Strength-of-Connection matrix
        Must have zero diagonal
    splitting : array
        C/F splitting stored in an array of length N
    theta : float in [0,1), default None
        theta value defining strong connections in a classical AMG
        sense. Provide if a different SOC is used for P than for
        CF-splitting; otherwise, theta = None.
    norm : string, default 'abs'
        Norm used in redefining classical SOC. Options are 'min' and
        'abs' for CSR matrices. See strength.py for more information.
    modified : bool, default True
        Use modified classical interpolation. More robust if RS coarsening with
        second pass is not used for CF splitting. Ignores interpolating from strong
        F-connections without a common C-neighbor.

    Returns
    -------
    P : csr_matrix
        Prolongator using classical interpolation; see Sec. 3 Eq. (8)
        of [0] for modified=False and Eq. (9) for modified=True.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.interpolate import classical_interpolation
    >>> import numpy as np
    >>> A = poisson((5,),format='csr')
    >>> splitting = np.array([1,0,1,0,1], dtype='intc')
    >>> P = classical_interpolation(A, A, splitting, 0.25)
    >>> print(P.todense())
    [[ 1.   0.   0. ]
     [ 0.5  0.5  0. ]
     [ 0.   1.   0. ]
     [ 0.   0.5  0.5]
     [ 0.   0.   1. ]]
    """
    from scipy.sparse import csr_matrix, isspmatrix_csr

    if not isspmatrix_csr(A):
        raise TypeError('expected csr_matrix for A')

    if not isspmatrix_csr(C):
        raise TypeError('Expected csr_matrix SOC matrix, C.')

    nc = np.sum(splitting)
    n = A.shape[0]

    if theta is not None:
        C = classical_strength_of_connection(A, theta=theta, norm=norm)
    else:
        C = C.copy()

    # Use modified classical interpolation by ignoring strong F-connections that do
    # not have a common C-point.
    if modified:
        amg_core_remove_strong_FF_connections(A.shape[0], C.indptr, C.indices,
                                              C.data, splitting)
    C.eliminate_zeros()

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, copy the entries of A into the
    # sparsity pattern of C.
    C.data[:] = 1.0
    C = C.multiply(A)

    P_indptr = np.empty_like(A.indptr)
    amg_core_rs_classical_interpolation_pass1(A.shape[0], C.indptr,
                                              C.indices, splitting, P_indptr)
    nnz = P_indptr[-1]
    P_indices = np.empty(nnz, dtype=P_indptr.dtype)
    P_data = np.empty(nnz, dtype=A.dtype)


    amg_core_rs_classical_interpolation_pass2(A.shape[0], A.indptr, A.indices,
                                              A.data, C.indptr, C.indices,
                                              C.data, splitting, P_indptr,
                                              P_indices, P_data, modified)

    P = csr_matrix((P_data, P_indices, P_indptr), shape=[n, nc])
    return csr_matrix((P_data, P_indices, P_indptr), shape=[n, nc])





# /*
#  *  Compute a strength of connection matrix using the classical strength
#  *  of connection measure by Ruge and Stuben. Both the input and output
#  *  matrices are stored in CSR format.  An off-diagonal nonzero entry
#  *  A[i,j] is considered strong if:
#  *
#  *  ..
#  *      |A[i,j]| >= theta * max( |A[i,k]| )   where k != i
#  *
#  * Otherwise, the connection is weak.
#  *
#  * Parameters
#  * ----------
#  * num_rows : int
#  *     number of rows in A
#  * theta : float
#  *     strength of connection tolerance
#  * Ap : array
#  *     CSR row pointer
#  * Aj : array
#  *     CSR index array
#  * Ax : array
#  *     CSR data array
#  * Sp : array
#  *     CSR row pointer
#  * Sj : array
#  *     CSR index array
#  * Sx : array
#  *     CSR data array
#  *
#  * Returns
#  * -------
#  * Nothing, S will be stored in Sp, Sj, Sx
#  *
#  * Notes
#  * -----
#  * Storage for S must be preallocated.  Since S will consist of a subset
#  * of A's nonzero values, a conservative bound is to allocate the same
#  * storage for S as is used by A.
#  */
# Reproduction of https://github.com/pyamg/pyamg/blob/e1fe54c93be1029c02ddcf84c2338a607b088703/pyamg/amg_core/ruge_stuben.h#L61
def amg_core_classical_strength_of_connection_abs(
        n_row: int,
        theta: float,
        Ap: np.ndarray,
        Aj: np.ndarray,
        Ax: np.ndarray,
        Sp: np.ndarray,
        Sj: np.ndarray,
        Sx: np.ndarray):
    nnz = 0
    Sp[0] = 0
    for i in range(n_row):
        max_offdiagonal = 0.0
        row_start = Ap[i]
        row_end = Ap[i+1]
        for jj in range(row_start, row_end):
            if Aj[jj] != i:
                max_offdiagonal = max(max_offdiagonal, np.abs(Ax[jj]))
        
        threshold = theta * max_offdiagonal
        for jj in range(row_start, row_end):
            norm_jj = np.abs(Ax[jj])

            # Add entry if it exceeds the threshold
            if norm_jj >= threshold:
                if Aj[jj] != i:
                    Sj[nnz] = Aj[jj]
                    Sx[nnz] = Ax[jj]
                    nnz += 1

            # Always add the diagonal
            if Aj[jj] == i:
                Sj[nnz] = Aj[jj]
                Sx[nnz] = Ax[jj]
                nnz += 1
            
        Sp[i+1] = nnz


# /* Compute a C/F (coarse-fine( splitting using the classical coarse grid
#  * selection method of Ruge and Stuben.  The strength of connection matrix S,
#  * and its transpose T, are stored in CSR format.  Upon return, the  splitting
#  * array will consist of zeros and ones, where C-nodes (coarse nodes) are
#  * marked with the value 1 and F-nodes (fine nodes) with the value 0.
#  *
#  * Parameters
#  * ----------
#  * n_nodes : int
#  *     number of rows in A
#  * Sp : array
#  *     CSR row pointer array for SOC matrix
#  * Sj : array
#  *     CSR column index array for SOC matrix
#  * Tp : array
#  *     CSR row pointer array for transpose of SOC matrix
#  * Tj : array
#  *     CSR column index array for transpose of SOC matrix
#  * influence : array
#  *     array that influences splitting (values stored here are
#  *     added to lambda for each point)
#  * splitting : array, inplace
#  *     array to store the C/F splitting
#  *
#  * Notes
#  * -----
#  * The splitting array must be preallocated
#  */
# from https://github.com/pyamg/pyamg/blob/e1fe54c93be1029c02ddcf84c2338a607b088703/pyamg/amg_core/ruge_stuben.h#L234
def amg_core_rs_cf_splitting(
        n_nodes: int,
        Sp: np.ndarray,
        Sj: np.ndarray,
        Tp: np.ndarray,
        Tj: np.ndarray,
        influence: np.ndarray,
        splitting: np.ndarray):
    #Fine node=1, Coarse node=0, Unmarked node=2
    F_NODE = 0
    C_NODE = 1
    U_NODE = 2
    PRE_F_NODE = 3

    lambda_ = np.zeros(n_nodes, dtype='intc')

    # Compute initial lambda based on C^T
    lambda_max = 0
    for i in range(n_nodes):
        lambda_[i] = Tp[i+1] - Tp[i] + influence[i]
        lambda_max = max(lambda_max, lambda_[i])

    # For each value of lambda, create an interval of nodes with that value
    lambda_max = lambda_max * 2
    if n_nodes + 1 > lambda_max:
        lambda_max = n_nodes + 1

    interval_ptr = np.zeros(lambda_max, dtype='intc')
    interval_count = np.zeros(lambda_max, dtype='intc')
    index_to_node = np.zeros(n_nodes, dtype='intc')
    node_to_index = np.zeros(n_nodes, dtype='intc')

    for i in range(n_nodes):
        interval_count[lambda_[i]] += 1
    cumsum = 0
    for i in range(lambda_max):
        interval_ptr[i] = cumsum
        cumsum += interval_count[i]
        interval_count[i] = 0
    for i in range(n_nodes):
        lambda_i = lambda_[i]
        index = interval_ptr[lambda_i] + interval_count[lambda_i]
        index_to_node[index] = i
        node_to_index[i] = index
        interval_count[lambda_i] += 1

    splitting.fill(U_NODE) #Fine node=1, Coarse node=0, Unmarked node=2

    # All nodes with no neighbors become F nodes
    for i in range(n_nodes):
        if lambda_[i] == 0 or (lambda_[i] == 1 and Tj[Tp[i]] == i):
            splitting[i] = 0

    # Add elements to C and F, in descending order of lambda
    for top_index in range(n_nodes - 1, -1, -1):
        i = index_to_node[top_index]
        lambda_i = lambda_[i]

        # Remove i from its interval
        interval_count[lambda_i] -= 1

        # If maximum lambda = 0, break out of loop
        if lambda_[i] <= 0:
            break

        # If node is unmarked, set maximum node as C-node and modify
        # lambda values in neighborhood
        if splitting[i] == U_NODE:
            splitting[i] = C_NODE

            # For each j in S^T_i /\ U, mark j as tentative F-point
            for jj in range(Tp[i], Tp[i+1]):
                j = Tj[jj]
                if splitting[j] == U_NODE:
                    splitting[j] = PRE_F_NODE

            # For each j in S^T_i /\ U marked as tentative F-point, modify lamdba
            # values for neighborhood of j
            for jj in range(Tp[i], Tp[i+1]):
                j = Tj[jj]
                if splitting[j] == PRE_F_NODE:
                    splitting[j] = F_NODE

                    # For each k in S_j /\ U, modify lambda value, lambda_k += 1
                    for kk in range(Sp[j], Sp[j+1]):
                        k = Sj[kk]

                        if splitting[k] == U_NODE:

                            # Move k to the end of its current interval
                            if lambda_[k] >= n_nodes - 1:
                                continue

                            lambda_k = lambda_[k]
                            old_pos = node_to_index[k]
                            new_pos = interval_ptr[lambda_k] + interval_count[lambda_k] - 1

                            node_to_index[index_to_node[old_pos]] = new_pos
                            node_to_index[index_to_node[new_pos]] = old_pos
                            index_to_node[old_pos], index_to_node[new_pos] = index_to_node[new_pos], index_to_node[old_pos]

                            # Update intervals
                            interval_count[lambda_k]   -= 1
                            interval_count[lambda_k+1] += 1
                            interval_ptr[lambda_k+1]    = new_pos
                            
                            # Increment lambda_k
                            lambda_[k] += 1

    # set any unmarked nodes as F-points
    for i in range(n_nodes):
        if splitting[i] == U_NODE:
            splitting[i] = F_NODE
 


# /* Remove strong F-to-F connections that do NOT have a common C-point from
#  * the set of strong connections. Specifically, set the data value in CSR
#  * format to 0. Removing zero entries afterwards will adjust row pointer
#  * and column indices.
#  *
#  * Parameters
#  * ----------
#  * n_nodes : int
#  *     Number of rows in A
#  * Sp : array
#  *     Row pointer for SOC matrix, C
#  * Sj : array
#  *     Column indices for SOC matrix, C
#  * Sx : array
#  *     Data array for SOC matrix, C
#  * splitting : array
#  *     Boolean array with 1 denoting C-points and 0 F-points
#  *
#  * Returns
#  * -------
#  * Nothing, Sx[] is set to zero to eliminate connections.
#  */
# from https://github.com/pyamg/pyamg/blob/e1fe54c93be1029c02ddcf84c2338a607b088703/pyamg/amg_core/ruge_stuben.h#L1014
def amg_core_remove_strong_FF_connections(n_nodes: int,
                                Sp: np.ndarray,
                                Sj: np.ndarray,
                                Sx: np.ndarray,
                                splitting: np.ndarray):
    F_NODE = 0
    C_NODE = 1
    for row in range(n_nodes):
        if splitting[row] == F_NODE:

            # For each j in S_row /\ F, test dependence of j on S_row /\ C
            for jj in range(Sp[row], Sp[row+1]):
                j = Sj[jj]
                if splitting[j] == F_NODE:

                    # Test dependence, i.e. check that S_j /\ S_row /\ C is
                    # nonempty. This is simply checking that nodes j and row
                    # have a common strong C-point connection.
                    dependence = False
                    for ii in range(Sp[row], Sp[row+1]):
                        row_ind = Sj[ii]
                        if splitting[row_ind] == C_NODE:
                            for kk in range(Sp[j], Sp[j+1]):
                                if Sj[kk] == row_ind:
                                    dependence = True
                                if dependence:
                                    break
                        if dependence:
                            break
                    
                    # Node j passed dependence test
                    if dependence:
                        continue
                    # Node j did not pass dependence test. That is, the two F-points
                    # do not have a common C neighbor, and we thus remove the strong
                    # connection.
                    else:
                        Sx[jj] = 0



# /* First pass of classical AMG interpolation to build row pointer for
#  * P based on SOC matrix and CF-splitting.
#  *
#  * Parameters
#  * ----------
#  * n_nodes : int
#  *     Number of rows in A
#  * Sp : array
#  *     Row pointer for SOC matrix, C
#  * Sj : array
#  *     Column indices for SOC matrix, C
#  * splitting : array
#  *     Boolean array with 1 denoting C-points and 0 F-points
#  * Pp : array
#  *     empty array to store row pointer for matrix P
#  *
#  * Returns
#  * -------
#  * Nothing, Pp is modified in place.
#  *
#  */
def amg_core_rs_classical_interpolation_pass1(n_nodes: int,
                                    Sp: np.ndarray,
                                    Sj: np.ndarray,
                                    splitting: np.ndarray,
                                    Pp: np.ndarray):
    C_NODE = 1

    nnz = 0
    Pp[0] = 0
    for i in range(n_nodes):
        if splitting[i] == C_NODE:
            nnz += 1
        else:
            for jj in range(Sp[i], Sp[i+1]):
                if (splitting[Sj[jj]] == C_NODE) and (Sj[jj] != i):
                    nnz += 1
        Pp[i+1] = nnz






# /* Produce a classical AMG interpolation operator for the case in which
#  * two strongly connected F -points do NOT have a common C-neighbor. Formula
#  * can be found in Sec. 3 Eq. (8) of [1] for modified=False and Eq. (9)
#  * for modified=True.
#  *
#  * Parameters
#  * ----------
#  * Ap : array
#  *     Row pointer for matrix A
#  * Aj : array
#  *     Column indices for matrix A
#  * Ax : array
#  *     Data array for matrix A
#  * Sp : array
#  *     Row pointer for SOC matrix, C
#  * Sj : array
#  *     Column indices for SOC matrix, C
#  * Sx : array
#  *     Data array for SOC matrix, C -- MUST HAVE VALUES OF A
#  * splitting : array
#  *     Boolean array with 1 denoting C-points and 0 F-points
#  * Pp : array
#  *     Row pointer for matrix P
#  * Pj : array
#  *     Column indices for matrix P
#  * Px : array
#  *     Data array for matrix P
#  * modified : bool
#  *     Use modified interpolation formula
#  *
#  * Notes
#  * -----
#  * For modified interpolation, it is assumed that SOC matrix C is
#  * passed in WITHOUT any F-to-F connections that do not share a
#  * common C-point neighbor. Any SOC matrix C can be set as such by
#  * calling remove_strong_FF_connections().
#  *
#  * Returns
#  * -------
#  * Nothing, Pj[] and Px[] modified in place.
#  *
#  * References
#  * ----------
#  * [0] V. E. Henson and U. M. Yang, BoomerAMG: a parallel algebraic multigrid
#  *      solver and preconditioner, Applied Numerical Mathematics 41 (2002).
#  *
#  * [1] "Distance-Two Interpolation for Parallel Algebraic Multigrid,"
#  *      H. De Sterck, R. Falgout, J. Nolting, U. M. Yang, (2008).
#  */
# from https://github.com/pyamg/pyamg/blob/e1fe54c93be1029c02ddcf84c2338a607b088703/pyamg/amg_core/ruge_stuben.h#L1115
def amg_core_rs_classical_interpolation_pass2(n_nodes: int,
                                    Ap: np.ndarray,
                                    Aj: np.ndarray,
                                    Ax: np.ndarray,
                                    Sp: np.ndarray,
                                    Sj: np.ndarray,
                                    Sx: np.ndarray,
                                    splitting: np.ndarray,
                                    Pp: np.ndarray,
                                    Pj: np.ndarray,
                                    Px: np.ndarray,
                                    modified: bool):
    C_NODE = 1
    F_NODE = 0

    for i in range(n_nodes):
        # If node i is a C-point, then set interpolation as injection
        if splitting[i] == C_NODE:
            Pj[Pp[i]] = i
            Px[Pp[i]] = 1

        # Otherwise, use RS classical interpolation formula
        else:
            # Calculate denominator
            denominator = 0

            # Start by summing entire row of A 
            for mm in range(Ap[i], Ap[i+1]):
                denominator += Ax[mm]
            
            # Then subtract off the strong connections so that you are left with
            # denominator = a_ii + sum_{m in weak connections} a_im
            for mm in range(Sp[i], Sp[i+1]):
                if Sj[mm] != i:
                    denominator -= Sx[mm] # making sure to leave the diagonal entry in there

            # Set entries in P (interpolation weights w_ij from strongly connected C-points)
            nnz = Pp[i]
            for jj in range(Sp[i], Sp[i+1]):
                if splitting[Sj[jj]] == C_NODE:
                    # Set temporary value for Pj as global index, j. Will be mapped to
                    # appropriate coarse-grid column index after all data is filled in.
                    Pj[nnz] = Sj[jj]
                    j = Sj[jj]

                    # Initialize numerator as a_ij
                    numerator = Sx[jj]

                    # Sum over strongly connected fine points
                    for kk in range(Sp[i], Sp[i+1]):
                        if (splitting[Sj[kk]] == F_NODE) and (Sj[kk] != i):
                            # Get column k and value a_ik
                            k = Sj[kk]
                            a_ik = Sx[kk]

                            # Get a_kj (have to search over k'th row in A for connection a_kj)
                            a_kj = 0
                            a_kk = 0

                            if(modified):
                                for search_ind in range(Ap[k], Ap[k+1]):
                                    if Aj[search_ind] == j:
                                        a_kj = Ax[search_ind]
                                    elif Aj[search_ind] == k:
                                        a_kk = Ax[search_ind]
                            else:
                                for search_ind in range(Ap[k], Ap[k+1]):
                                    if Aj[search_ind] == j:
                                        a_kj = Ax[search_ind]
                                        break

                            # If sign of a_kj matches sign of a_kk, ignore a_kj in sum
                            # (i.e. leave as a_kj = 0) for modified interpolation
                            if(modified and a_kj * a_kk > 0):
                                a_kj = 0

                            # If a_kj == 0, then we don't need to do any more work, otherwise
                            # proceed to account for node k's contribution
                            if(abs(a_kj) > 1e-15 * abs(a_ik)):

                                # Calculate sum for inner denominator (loop over strongly connected C-points)
                                inner_denominator = 0
                                for ll in range(Sp[i], Sp[i+1]):
                                    if splitting[Sj[ll]] == C_NODE:
                                        # Get column l
                                        l = Sj[ll]

                                        # Add connection a_kl if present in matrix (search over kth row in A for connection)
                                        # Only add if sign of a_kl does not equal sign of a_kk
                                        for search_ind in range(Ap[k], Ap[k+1]):
                                            if Aj[search_ind] == l:
                                                a_kl = Ax[search_ind]
                                                if (not modified) or (np.sign(a_kl) != np.sign(a_kk)):
                                                    inner_denominator += a_kl
                                                break
                                
                                # Add a_ik * a_kj / inner_denominator to the numerator
                                if abs(inner_denominator) < 1e-15 * abs(a_ik * a_kj):
                                    print("Inner denominator was zero.")
                                numerator += a_ik * a_kj / inner_denominator
                
                    # Set w_ij = -numerator/denominator
                    if abs(denominator) < 1e-15 * abs(numerator):
                        print("Outer denominator was zero: diagonal plus sum of weak connections was zero.")
                    Px[nnz] = -numerator / denominator
                    nnz += 1

    # Column indices were initially stored as global indices. Build map to switch
    # to C-point indices.
    map = np.zeros(n_nodes, dtype='intc')
    
    sum = 0
    for i in range(n_nodes):
        map[i] = sum
        sum += splitting[i]
    for i in range(Pp[n_nodes]):
        Pj[i] = map[Pj[i]]

    return
                

# judge if A is positive definite
# https://stackoverflow.com/a/44287862/19253199
# if A is symmetric and able to be Cholesky decomposed, then A is positive definite
def is_spd(A):
    A=A.toarray()
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

def generate_A_b_spd(n=1000):
    import scipy.sparse as sp
    A = sp.random(n, n, density=0.01, format="csr")
    A = A.T @ A
    b = np.random.rand(A.shape[0])
    flag = is_spd(A)
    print(f"is_spd: {flag}")
    print(f"Generated spd A: {A.shape}, b: {b.shape}")
    A = sp.csr_matrix(A)
    return A, b

if __name__ == '__main__':
    # test_classical_strength_of_connection()


    # A = poisson((100,)).tocsr()
    A,_ = generate_A_b_spd(100)

    print("A\n",A.A)

    print("my version")
    P1,sp1 = main(A)

    print("\n\npyamg version")
    P2,sp2 = test_pyamg(A)
    
    assert np.allclose(sp1, sp2) 
    assert np.allclose(P1.toarray(), P2.toarray())
    print("Same as pyamg")

