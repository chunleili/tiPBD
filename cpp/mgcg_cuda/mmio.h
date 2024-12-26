#pragma once
#include "SpMatData.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

template<typename T=std::vector<float>>
void savetxt(T &field, std::string filename)
{
    std::ofstream myfile;
    myfile.open(filename);
    for(auto &i:field)
    {
        myfile << i << '\n';
    }
    myfile.close();
}

template<typename T=FieldXi>
void savetxt2d(T &field, std::string filename="1.txt")
{
    std::ofstream myfile;
    myfile.open(filename);
    for(auto &i:field)
    {
        for(auto &ii:i)
        {
            myfile << ii << " ";
        }
        myfile << std::endl;
    }
    myfile.close();
}


template<typename T=FieldXi>
void loadtxt(T &M, std::string filename)
{
  std::ifstream inputFile(filename);
  std::string line;

  unsigned int rows = 0;
  while (std::getline(inputFile, line))
  {
    std::istringstream iss(line);
    int val;
    M.resize(rows + 1);
    while (iss >> val)
    {
      M[rows].push_back(val);
    }
    rows++;
  }
}




template <typename T=float>
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

template <typename T=float>
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

