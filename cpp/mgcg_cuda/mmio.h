#pragma once
#include "SpMatData.h"


void mmwrite(SpMatData* A, std::string filename="mat.mtx")
{
    std::ofstream myfile;
    myfile.open(filename);
    std::vector<float>& data = A->data;
    std::vector<int>& indices = A->indices;
    std::vector<int>& indptr = A->indptr;
    int nrows = A->nrows();
    int ncols = A->ncols();
    int nnz = A->nnz();
    myfile<<"%%MatrixMarket matrix coordinate  real general\n";
    myfile<<nrows<<" "<<ncols<<" "<<nnz<<"\n";
    for(int i=0; i<nrows; i++)
    {
        for(int j=indptr[i]; j<indptr[i+1]; j++)
        {
            myfile << i+1 << " " << indices[j]+1 << " " << data[j] << '\n';
        }
    }
    myfile.close();
}


void mmread(SpMatData* A, std::string filename="mat.mtx")
{
    std::ifstream myfile;
    myfile.open(filename);
    std::string line;
    std::getline(myfile, line);
    std::getline(myfile, line);
    std::istringstream iss(line);
    int nrows, ncols, nnz;
    iss >> nrows >> ncols >> nnz;
    // A->nrows = nrows;
    // A->ncols = ncols;
    // A->nnz = nnz;
    A->indptr.resize(nrows+1);
    A->indices.resize(nnz);
    A->data.resize(nnz);
    for(int i=0; i<nrows; i++)
    {
        A->indptr[i] = 0;
    }
    A->indptr[nrows] = nnz;
    for(int i=0; i<nnz; i++)
    {
        int row, col;
        float val;
        myfile >> row >> col >> val;
        A->indptr[row-1]++;
        A->indices[i] = col-1;
        A->data[i] = val;
    }
    for(int i=1; i<=nrows; i++)
    {
        A->indptr[i] += A->indptr[i-1];
    }
    myfile.close();
}

