#pragma once
#include <vector>

struct SpMatData
{
    std::vector<float> data; //non-zero values
    std::vector<int> indices; //column indices
    std::vector<int> indptr;    //row starts
    std::vector<int> ii;//this is redundant for CSR, but (ii, indicices, data) can be used as COO format
    int nrows() const { return indptr.size() - 1; }
    int ncols() const { return nrows(); }
    int nnz() const { return data.size(); }
    void operator=(const SpMatData& other)
    {
        data = other.data;
        indices = other.indices;
        indptr = other.indptr;
        ii = other.ii;
    }
};
