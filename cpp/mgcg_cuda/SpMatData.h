#pragma once
#include <vector>

struct SpMatData
{
    std::vector<float> data;
    std::vector<int> indices;
    std::vector<int> indptr;
    int nrows() const { return indptr.size() - 1; }
    int ncols() const { return nrows(); }
    int nnz() const { return data.size(); }
};
