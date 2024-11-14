#include "smoothed_aggregation.h"

using std::string;
using std::vector;

struct CSRCpp;
void transposeImply(CSRCpp& A, CSRCpp& AT)
{

}

struct CSRCpp {
    std::vector<int> indices;
    std::vector<float> data;
    std::vector<int> indptr;
    int64_t nrows;
    int64_t ncols;
    int64_t numnonz;

    CSRCpp() noexcept : nrows(0), ncols(0), numnonz(0) {}


    void resize(size_t rows, size_t cols, size_t nnz) {
        nrows = rows;
        ncols = cols;
        numnonz = nnz;
        data.resize(nnz);
        indices.resize(nnz);
        indptr.resize(rows + 1);
    }

    CSRCpp transpose(){
        CSRCpp AT;
        transposeImply(*this, AT);
        return AT;
    }
};

struct Level
{
    CSRCpp A;
    std::vector<float> B;
    CSRCpp C;
    CSRCpp AggOp;
    CSRCpp T;
    CSRCpp P;
    CSRCpp R;
};

struct MultilevelSolver
{
    std::vector<Level> levels;
    MultilevelSolver(std::vector<Level> levels) : levels(levels) {}
};


void vecabs(std::vector<float>& vec)
{
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = std::fabs(vec[i]);
    }
}

void scale_rows_by_largest_entry(CSRCpp& A)
{
    for (int i = 0; i < A.nrows; i++) {
        float max_val = 0;
        for (int j = A.indptr[i]; j < A.indptr[i + 1]; j++) {
            max_val = std::max(max_val, std::fabs(A.data[j]));
        }
        for (int j = A.indptr[i]; j < A.indptr[i + 1]; j++) {
            if (max_val != 0)
                A.data[j] /= max_val;
        }
    }
}

void symmetric_strength_of_connection(CSRCpp& A, CSRCpp& C, float theta=0.25)
{
    symmetric_strength_of_connection<int, float, float>(A.nrows, theta, A.indptr.data(), A.indptr.size(), A.indices.data(), A.indices.size(), A.data.data(), A.data.size(), C.indptr.data(), C.indptr.size(), C.indices.data(), C.indices.size(), C.data.data(), C.data.size());
    vecabs(C.data);
    scale_rows_by_largest_entry(C);
}


void standard_aggregation(CSRCpp& C, CSRCpp& AggOp)
{

    int num_rows = C.nrows;

    std::vector<int> Tj(num_rows);  // stores the aggregate #s
    std::vector<int> Cpts(num_rows);  // stores the Cpts

    int num_aggregates = standard_aggregation<int>(num_rows, C.indptr.data(), C.indptr.size(), C.indices.data(), C.indices.size(), Tj.data(), Tj.size(), Cpts.data(), Cpts.size());
    Cpts.resize(num_aggregates);

    // no nodes aggregated
    if (num_aggregates == 0) {
        // return all zero matrix and no Cpts
        AggOp.resize(num_rows, 1, 0);
        return;
    }

    std::vector<int> shape = {num_rows, num_aggregates};


    // all nodes aggregated
    std::vector<int> Tp(num_rows + 1);
    for (int i = 0; i < num_rows + 1; i++) {
        Tp[i] = i;
    }
    std::vector<float> Tx(Tj.size(), 1);
    AggOp.resize(shape[0], shape[1], Tx.size());
    AggOp.indptr = Tp;
    AggOp.indices = Tj;
    AggOp.data = Tx;

    return;
}

struct BSRCpp
{
    std::vector<float> data;
    std::vector<int> indices;
    std::vector<int> indptr;
    int64_t nrows;
    int64_t ncols;
    int64_t blocksize;

    BSRCpp() noexcept : nrows(0), ncols(0), blocksize(0) {}

    void resize(size_t rows, size_t cols, size_t blocksize, size_t nnz) {
        nrows = rows;
        ncols = cols;
        blocksize = blocksize;
        data.resize(nnz);
        indices.resize(nnz);
        indptr.resize(rows + 1);
    }
};


// QR decomposition, Q as new P, R as new B 
void  fit_candidates(CSRCpp& AggOp, std::vector<float>& B, CSRCpp& Q, std::vector<float>& R, float tol = 1e-10)
{
    // N_fine, N_coarse = AggOp.shape
    int N_fine = AggOp.nrows;
    int N_coarse = AggOp.ncols;
    
    int K1 = B.size() / N_fine;  // dof per supernode (e.g. 3 for 3d vectors)
    int K2 = 1;                // candidates

    // the first two dimensions of R and Qx are collapsed later
    R.resize(N_coarse* K2* K2);
    std::vector<float> Qx (AggOp.numnonz * K1 * K2);  // BSR data array

    CSRCpp AggOp_csc = AggOp.transpose();

    ;
    fit_candidates_real<int, float>(N_fine, N_coarse, K1, K2, AggOp_csc.indptr.data(), AggOp_csc.indptr.size(), AggOp_csc.indices.data(), AggOp_csc.indices.size(), Qx.data(), Qx.size(), B.data(), B.size(), R.data(), R.size(), tol);

    // FIXME
    // Q = bsr_matrix((Qx.swapaxes(1, 2).copy(), AggOp_csc.indices,
    //                 AggOp_csc.indptr), shape=(K2*N_coarse, K1*N_fine))
    // Q = Q.T.tobsr()
    // R = R.reshape(-1, K2)
}

CSRCpp spmatXspmatImply(CSRCpp& A, CSRCpp& B)
{
    CSRCpp C;
    return C;
}

CSRCpp RAPImply(CSRCpp& A, CSRCpp& R, CSRCpp& P)
{
    CSRCpp RA = spmatXspmatImply(R, A);
    CSRCpp RAP = spmatXspmatImply(RA, P);
    return RAP;
}


void _extend_hierarchy(std::vector<Level>& levels, std::string strength, std::string aggregate, std::string smooth, std::string improve_candidates, bool diagonal_dominance, bool keep)
{
    CSRCpp& A = levels.back().A;
    std::vector<float>& B = levels.back().B;
    CSRCpp& C = levels.back().C;
    CSRCpp& AggOp = levels.back().AggOp;
    CSRCpp& T = levels.back().T;
    CSRCpp& P = levels.back().P;
    CSRCpp& R = levels.back().R;

    symmetric_strength_of_connection(A, C);
    standard_aggregation(C, AggOp);
    std::vector<float> newB;
    fit_candidates(AggOp, B, T, newB);

    levels.back().P = T;
    levels.back().R = T.transpose();

    levels.push_back(Level());
    A = RAPImply(A, levels.back().R, levels.back().P);
    levels.back().A = A;
    levels.back().B = newB;
}

MultilevelSolver smoothed_aggregation_solver(CSRCpp A, std::vector<float> B, CSRCpp BH, std::string symmetry, std::string strength, std::string aggregate, std::string smooth, std::string presmoother, std::string postsmoother, std::string improve_candidates, int max_levels, int max_coarse, bool diagonal_dominance, bool keep)
{
    if (symmetry != "symmetric" && symmetry != "hermitian" && symmetry != "nonsymmetric") {
        throw std::invalid_argument("Expected 'symmetric', 'nonsymmetric' or 'hermitian' for the symmetry parameter");
    }
    if (A.nrows != A.ncols) {
        throw std::invalid_argument("expected square matrix");
    }

    // Construct multilevel structure
    std::vector<Level> levels;
    levels.push_back(Level());
    levels.back().A = A;          // matrix

    // Append near nullspace candidates
    levels.back().B = B;          // right candidates


    while (levels.size() < max_levels && int(levels.back().A.nrows) > max_coarse) {
        _extend_hierarchy(levels, strength, aggregate, smooth, improve_candidates, diagonal_dominance, keep);
    }

    MultilevelSolver ml(levels);
    return ml;
}
