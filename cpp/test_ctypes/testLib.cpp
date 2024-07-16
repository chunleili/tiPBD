#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <iostream>
using namespace std;

extern "C"  DLLEXPORT int add_one(int i)
{
    return i+1;
}

extern "C" DLLEXPORT void print_array(double* array, int N)
{
    for (int i=0; i<N; i++) 
        cout << i << " " << array[i] << endl;
}


extern "C" DLLEXPORT void change_array(double* array, int N)
{
    for (int i=0; i<N; i++) 
        array[i] += + 1;
}



extern "C" DLLEXPORT void change_spmat(int* indptr, int* indices, double* data, int nrows, int ncols, int nnz)
{
    for (int i=0; i<nrows; i++) 
        for (int j=indptr[i]; j<indptr[i+1]; j++)
            data[j] += 1;
}


extern "C" DLLEXPORT void change_scalar(int* val)
{
    val[0] += 1;
}