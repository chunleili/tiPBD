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


def main():
    A = poisson((4,)).tocsr()
    print(A.toarray())
    levels = ruge_stuben_solver_my(A, max_levels=2, max_coarse=2)


def test_pyamg():
    A = poisson((4,)).tocsr()
    print(A.toarray())
    ml = pyamg.ruge_stuben_solver(A, max_levels=2, max_coarse=2, CF=('RS', {'second_pass': False}))
    P = ml.levels[0].P
    R = ml.levels[0].R
    print(P.toarray())
    r = []
    x = ml.solve(np.ones(A.shape[0]), tol=1e-3, residuals=r, maxiter=1)
    print(r)


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
        if bottom:
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


# /*
#  *  Compute a strength of connection matrix using the classical strength
#  *  of connection measure by Ruge and Stuben. Both the input and output
#  *  matrices are stored in CSR format.  An off-diagonal nonzero entry
#  *  A[i,j] is considered strong if:
#  *
#  *  ..
#  *      |A[i,j]| >= theta * max( |A[i,k]| )   where k != i
#  *
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
# from https://github.com/pyamg/pyamg/blob/e1fe54c93be1029c02ddcf84c2338a607b088703/pyamg/amg_core/ruge_stuben.h#L234
def amg_core_rs_cf_splitting(
        n_nodes: int,
        Sp: np.ndarray,
        Sj: np.ndarray,
        Tp: np.ndarray,
        Tj: np.ndarray,
        influence: np.ndarray,
        splitting: np.ndarray):
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

    splitting.fill(2)

    # All nodes with no neighbors become F nodes
    for i in range(n_nodes):
        if lambda_[i] == 0 or (lambda_[i] == 1 and Tj[Tp[i]] == i):
            splitting[i] = 0

    # Add elements to C and F, in descending order of lambda
    top_index = n_nodes - 1
    while top_index > -1:
        i = index_to_node[top_index]
        lambda_i = lambda_[i]

        # Remove i from its interval
        interval_count[lambda_i] -= 1

        # If maximum lambda = 0, break out of loop
        if lambda_[i] <= 0:
            break
 


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
# template<class I>
# void rs_cf_splitting(const I n_nodes,
#                      const I Sp[], const int Sp_size,
#                      const I Sj[], const int Sj_size,
#                      const I Tp[], const int Tp_size,
#                      const I Tj[], const int Tj_size,
#                      const I influence[], const int influence_size,
#                            I splitting[], const int splitting_size)
# {
#     std::vector<I> lambda(n_nodes,0);

#     // Compute initial lambda based on C^T
#     I lambda_max = 0;
#     for (I i = 0; i < n_nodes; i++) {
#         lambda[i] = Tp[i+1] - Tp[i] + influence[i];
#         if (lambda[i] > lambda_max) {
#             lambda_max = lambda[i];
#         }
#     }

#     // For each value of lambda, create an interval of nodes with that value
#     //      interval_ptr - the first index of the interval
#     //      interval_count - the number of indices in that interval
#     //      index_to_node - the node located at a given index
#     //      node_to_index - the index of a given node
#     lambda_max = lambda_max*2;
#     if (n_nodes+1 > lambda_max) {
#         lambda_max = n_nodes+1;
#     }

#     std::vector<I> interval_ptr(lambda_max,0);
#     std::vector<I> interval_count(lambda_max,0);
#     std::vector<I> index_to_node(n_nodes);
#     std::vector<I> node_to_index(n_nodes);

#     for (I i = 0; i < n_nodes; i++) {
#         interval_count[lambda[i]]++;
#     }
#     for (I i = 0, cumsum = 0; i < lambda_max; i++) {
#         interval_ptr[i] = cumsum;
#         cumsum += interval_count[i];
#         interval_count[i] = 0;
#     }
#     for (I i = 0; i < n_nodes; i++) {
#         I lambda_i = lambda[i];

#         I index    = interval_ptr[lambda_i] + interval_count[lambda_i];
#         index_to_node[index] = i;
#         node_to_index[i]     = index;
#         interval_count[lambda_i]++;
#     }

#     std::fill(splitting, splitting + n_nodes, U_NODE);

#     // All nodes with no neighbors become F nodes
#     for (I i = 0; i < n_nodes; i++) {
#         if (lambda[i] == 0 || (lambda[i] == 1 && Tj[Tp[i]] == i))
#             splitting[i] = F_NODE;
#     }

#     // Add elements to C and F, in descending order of lambda
#     for (I top_index=(n_nodes - 1); top_index>-1; top_index--) {

#         I i        = index_to_node[top_index];
#         I lambda_i = lambda[i];

#         // Remove i from its interval
#         interval_count[lambda_i]--;

#         // ----------------- Sorting every iteration = O(n^2) complexity ----------------- //
#         // Search over this interval to make sure we process nodes in descending node order
#         // I max_node = i;
#         // I max_index = top_index;
#         // for (I j = interval_ptr[lambda_i]; j < interval_ptr[lambda_i] + interval_count[lambda_i]; j++) {
#         //     if (index_to_node[j] > max_node) {
#         //         max_node = index_to_node[j];
#         //         max_index = j;
#         //     }
#         // }
#         // node_to_index[index_to_node[top_index]] = max_index;
#         // node_to_index[index_to_node[max_index]] = top_index;
#         // std::swap(index_to_node[top_index], index_to_node[max_index]);
#         // i = index_to_node[top_index];

#         // If maximum lambda = 0, break out of loop
#         if (lambda[i] <= 0) {
#             break;
#         }

#         // If node is unmarked, set maximum node as C-node and modify
#         // lambda values in neighborhood
#         if ( splitting[i] == U_NODE) {
#             splitting[i] = C_NODE;

#             // For each j in S^T_i /\ U, mark j as tentative F-point
#             for (I jj = Tp[i]; jj < Tp[i+1]; jj++) {
#                 I j = Tj[jj];
#                 if(splitting[j] == U_NODE) {
#                     splitting[j] = PRE_F_NODE;
#                 }
#             }

#             // For each j in S^T_i /\ U marked as tentative F-point, modify lamdba
#             // values for neighborhood of j
#             for (I jj = Tp[i]; jj < Tp[i+1]; jj++)
#             {
#                 I j = Tj[jj];
#                 if(splitting[j] == PRE_F_NODE)
#                 {
#                     splitting[j] = F_NODE;

#                     // For each k in S_j /\ U, modify lambda value, lambda_k += 1
#                     for (I kk = Sp[j]; kk < Sp[j+1]; kk++){
#                         I k = Sj[kk];

#                         if(splitting[k] == U_NODE){

#                             // Move k to the end of its current interval
#                             if(lambda[k] >= n_nodes - 1) {
#                                 continue;
#                             }

#                             I lambda_k = lambda[k];
#                             I old_pos  = node_to_index[k];
#                             I new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k] - 1;

#                             node_to_index[index_to_node[old_pos]] = new_pos;
#                             node_to_index[index_to_node[new_pos]] = old_pos;
#                             std::swap(index_to_node[old_pos], index_to_node[new_pos]);

#                             // Update intervals
#                             interval_count[lambda_k]   -= 1;
#                             interval_count[lambda_k+1] += 1; //invalid write!
#                             interval_ptr[lambda_k+1]    = new_pos;

#                             // Increment lambda_k
#                             lambda[k]++;
#                         }
#                     }
#                 }
#             }

#             // For each j in S_i /\ U, set lambda_j -= 1
#             for (I jj = Sp[i]; jj < Sp[i+1]; jj++) {
#                 I j = Sj[jj];
#                 // Decrement lambda for node j
#                 if (splitting[j] == U_NODE) {
#                     if (lambda[j] == 0) {
#                         continue;
#                     }

#                     // Move j to the beginning of its current interval
#                     I lambda_j = lambda[j];
#                     I old_pos  = node_to_index[j];
#                     I new_pos  = interval_ptr[lambda_j];

#                     node_to_index[index_to_node[old_pos]] = new_pos;
#                     node_to_index[index_to_node[new_pos]] = old_pos;
#                     std::swap(index_to_node[old_pos],index_to_node[new_pos]);

#                     // Update intervals
#                     interval_count[lambda_j]   -= 1;
#                     interval_count[lambda_j-1] += 1;
#                     interval_ptr[lambda_j]     += 1;
#                     interval_ptr[lambda_j-1]    = interval_ptr[lambda_j] - interval_count[lambda_j-1];

#                     // Decrement lambda_j
#                     lambda[j]--;
#                 }
#             }
#         }
#     }

#     // Set any unmarked nodes as F-points
#     for (I i=0; i<n_nodes; i++) {
#         if (splitting[i] == U_NODE) {
#             splitting[i] = F_NODE;
#         }
#     }
# }





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
# template<class I, class T>
# void remove_strong_FF_connections(const I n_nodes,
#                                   const I Sp[], const int Sp_size,
#                                   const I Sj[], const int Sj_size,
#                                         T Sx[], const int Sx_size,
#                                   const I splitting[], const int splitting_size)
# {
#     // For each F-point
#     for (I row=0; row<n_nodes; row++) {
#         if (splitting[row] == F_NODE) {

#             // For each j in S_row /\ F, test dependence of j on S_row /\ C
#             for (I jj=Sp[row]; jj<Sp[row+1]; jj++) {
#                 I j = Sj[jj];

#                 if (splitting[j] == F_NODE) {

#                     // Test dependence, i.e. check that S_j /\ S_row /\ C is
#                     // nonempty. This is simply checking that nodes j and row
#                     // have a common strong C-point connection.
#                     bool dependence = false;
#                     for (I ii=Sp[row]; ii<Sp[row+1]; ii++) {
#                         I row_ind = Sj[ii];
#                         if (splitting[row_ind] == C_NODE) {
#                             for (I kk=Sp[j]; kk<Sp[j+1]; kk++) {
#                                 if (Sj[kk] == row_ind) {
#                                     dependence = true;
#                                 }
#                             }
#                         }
#                         if (dependence) {
#                             break;
#                         }
#                     }

#                     // Node j passed dependence test
#                     if (dependence) {
#                         continue;
#                     }
#                     // Node j did not pass dependence test. That is, the two F-points
#                     // do not have a common C neighbor, and we thus remove the strong
#                     // connection.
#                     else {
#                         Sx[jj] = 0;
#                     }
#                 }
#             }
#         }
#     }
# }








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
# template<class I>
# void rs_classical_interpolation_pass1(const I n_nodes,
#                                       const I Sp[], const int Sp_size,
#                                       const I Sj[], const int Sj_size,
#                                       const I splitting[], const int splitting_size,
#                                             I Pp[], const int Pp_size)
# {
#     I nnz = 0;
#     Pp[0] = 0;
#     for (I i = 0; i < n_nodes; i++){
#         if( splitting[i] == C_NODE ){
#             nnz++;
#         }
#         else {
#             for (I jj = Sp[i]; jj < Sp[i+1]; jj++){
#                 if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) )
#                     nnz++;
#             }
#         }
#         Pp[i+1] = nnz;
#     }
# }













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
# template<class I, class T>
# void rs_classical_interpolation_pass2(const I n_nodes,
#                                       const I Ap[], const int Ap_size,
#                                       const I Aj[], const int Aj_size,
#                                       const T Ax[], const int Ax_size,
#                                       const I Sp[], const int Sp_size,
#                                       const I Sj[], const int Sj_size,
#                                       const T Sx[], const int Sx_size,
#                                       const I splitting[], const int splitting_size,
#                                       const I Pp[], const int Pp_size,
#                                             I Pj[], const int Pj_size,
#                                             T Px[], const int Px_size,
#                                       const bool modified)
# {
#     for (I i = 0; i < n_nodes; i++) {
#         // If node i is a C-point, then set interpolation as injection
#         if(splitting[i] == C_NODE) {
#             Pj[Pp[i]] = i;
#             Px[Pp[i]] = 1;
#         }
#         // Otherwise, use RS classical interpolation formula
#         else {

#             // Calculate denominator
#             T denominator = 0;

#             // Start by summing entire row of A
#             for (I mm = Ap[i]; mm < Ap[i+1]; mm++) {
#                 denominator += Ax[mm];
#             }

#             // Then subtract off the strong connections so that you are left with
#             // denominator = a_ii + sum_{m in weak connections} a_im
#             for (I mm = Sp[i]; mm < Sp[i+1]; mm++) {
#                 if ( Sj[mm] != i ) {
#                     denominator -= Sx[mm]; // making sure to leave the diagonal entry in there
#                 }
#             }

#             // Set entries in P (interpolation weights w_ij from strongly connected C-points)
#             I nnz = Pp[i];
#             for (I jj = Sp[i]; jj < Sp[i+1]; jj++) {

#                 if (splitting[Sj[jj]] == C_NODE) {

#                     // Set temporary value for Pj as global index, j. Will be mapped to
#                     // appropriate coarse-grid column index after all data is filled in.
#                     Pj[nnz] = Sj[jj];
#                     I j = Sj[jj];

#                     // Initialize numerator as a_ij
#                     T numerator = Sx[jj];

#                     // Sum over strongly connected fine points
#                     for (I kk = Sp[i]; kk < Sp[i+1]; kk++) {
#                         if ( (splitting[Sj[kk]] == F_NODE) && (Sj[kk] != i) ) {

#                             // Get column k and value a_ik
#                             I k = Sj[kk];
#                             T a_ik = Sx[kk];

#                             // Get a_kj (have to search over k'th row in A for connection a_kj)
#                             T a_kj = 0;
#                             T a_kk = 0;
#                             if (modified) {
#                                 for (I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
#                                     if (Aj[search_ind] == j) {
#                                         a_kj = Ax[search_ind];
#                                     }
#                                     else if (Aj[search_ind] == k) {
#                                         a_kk = Ax[search_ind];
#                                     }
#                                 }
#                             }
#                             else {
#                                 for (I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
#                                     if ( Aj[search_ind] == j ) {
#                                         a_kj = Ax[search_ind];
#                                         break;
#                                     }
#                                 }
#                             }

#                             // If sign of a_kj matches sign of a_kk, ignore a_kj in sum
#                             // (i.e. leave as a_kj = 0) for modified interpolation
#                             if ( modified && (signof(a_kj) == signof(a_kk)) ) {
#                                 a_kj = 0;
#                             }

#                             // If a_kj == 0, then we don't need to do any more work, otherwise
#                             // proceed to account for node k's contribution
#                             if (std::abs(a_kj) > 1e-15*std::abs(a_ik)) {

#                                 // Calculate sum for inner denominator (loop over strongly connected C-points)
#                                 T inner_denominator = 0;
#                                 for (I ll = Sp[i]; ll < Sp[i+1]; ll++) {
#                                     if (splitting[Sj[ll]] == C_NODE) {

#                                         // Get column l
#                                         I l = Sj[ll];

#                                         // Add connection a_kl if present in matrix (search over kth row in A for connection)
#                                         // Only add if sign of a_kl does not equal sign of a_kk
#                                         for (I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
#                                             if (Aj[search_ind] == l) {
#                                                 T a_kl = Ax[search_ind];
#                                                 if ( (!modified) || (signof(a_kl) != signof(a_kk)) ) {
#                                                     inner_denominator += a_kl;
#                                                 }
#                                                 break;
#                                             }
#                                         }
#                                     }
#                                 }

#                                 // Add a_ik * a_kj / inner_denominator to the numerator
#                                 if (std::abs(inner_denominator) < 1e-15*std::abs(a_ik * a_kj)) {
#                                     printf("Inner denominator was zero.\n");
#                                 }
#                                 numerator += a_ik * a_kj / inner_denominator;
#                             }
#                         }
#                     }

#                     // Set w_ij = -numerator/denominator
#                     if (std::abs(denominator) < 1e-15*std::abs(numerator)) {
#                         printf("Outer denominator was zero: diagonal plus sum of weak connections was zero.\n");
#                     }
#                     Px[nnz] = -numerator / denominator;
#                     nnz++;
#                 }
#             }
#         }
#     }

#     // Column indices were initially stored as global indices. Build map to switch
#     // to C-point indices.
#     std::vector<I> map(n_nodes);
#     for (I i = 0, sum = 0; i < n_nodes; i++) {
#         map[i]  = sum;
#         sum    += splitting[i];
#     }
#     for (I i = 0; i < Pp[n_nodes]; i++) {
#         Pj[i] = map[Pj[i]];
#     }
# }



if __name__ == '__main__':
    # main()
    test_pyamg()
    # test_classical_strength_of_connection()