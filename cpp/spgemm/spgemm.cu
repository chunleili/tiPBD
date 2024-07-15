// https://github.com/NVIDIA/CUDALibrarySamples/blob/ade391a17672d26e55429035450bc44afd277d34/cuSPARSE/spgemm/spgemm_example.c#L161
// https://docs.nvidia.com/cuda/cusparse/#cusparsespgemm
// https://github.com/NVIDIA/CUDALibrarySamples/tree/ade391a17672d26e55429035450bc44afd277d34/cuSPARSE/spgemm
// C = A * B
//--------------------------------------------------------
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <chrono>  
#include <string>          
// #include "eigen/unsupported/Eigen/SparseExtra" // Eigen::loadMarket

using std::string;

#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


#if __GNUC__ && __linux__
#include <sys/ptrace.h>

[[noreturn]] static void cuerr() {
    if (ptrace(PTRACE_TRACEME, 0, NULL, NULL) != 0)
        __builtin_trap();
    exit(EXIT_FAILURE);
}
#elif _WIN32 && _MSC_VER
#include <windows.h>

[[noreturn]] static void cuerr() {
    int debugger_present = 0;
    HANDLE process = GetCurrentProcess();
    CheckRemoteDebuggerPresent(process, &debugger_present);
    if (debugger_present) {
        __debugbreak();
    }
    exit(EXIT_FAILURE);
}
#else
[[noreturn]] static void cuerr() {
    exit(EXIT_FAILURE);
}
#endif



#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


// void assign(T const *datap, size_t ndat, int const *indicesp, size_t nind, int const *indptrp, size_t nptr, size_t rows, size_t cols, size_t nnz) {
//     data.resize(ndat);
//     CHECK_CUDA(cudaMemcpy(data.data(), datap, data.size() * sizeof(T), cudaMemcpyHostToDevice));
//     indices.resize(nind);
//     CHECK_CUDA(cudaMemcpy(indices.data(), indicesp, indices.size() * sizeof(int), cudaMemcpyHostToDevice));
//     indptr.resize(nptr);
//     CHECK_CUDA(cudaMemcpy(indptr.data(), indptrp, indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
//     nrows = rows;
//     ncols = cols;
//     numnonz = nnz;
// }

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <typename T=int>
std::vector<T> readTxt(std::string filename) {
    std::ifstream file(filename);
    std::vector<T> array;
    T value;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }

    while (file >> value) {
        array.push_back(value);
    }

    file.close();

    // for (int i : array) {
    //     std::cout << i << std::endl;
    // }
    std::cout<<filename<<" read successfully"<<std::endl;

    return array;
}


void readInfo(int &nrows, int &ncols, int &nnz, std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
    file >> nrows;
    file >> ncols;
    file >> nnz;
    file.close();
    std::cout<<"nrows: "<<nrows<<" ncols: "<<ncols<<" nnz: "<<nnz<<std::endl;
}

void readCSR(std::string filename, std::vector<int>& hA_csrOffsets, std::vector<int>& hA_columns, std::vector<float>& hA_values) {
    // auto indptr = readTxt(filename+"indptr.txt");
    // auto indices = readTxt(filename+"indices.txt");
    // auto data = readTxt<float>(filename+"data.txt");
    hA_csrOffsets = readTxt<int>(filename+"indptr.txt");
    hA_columns = readTxt<int>(filename+"indices.txt");
    hA_values = readTxt<float>(filename+"data.txt");
}

void printArr(int *values, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;
}

template<typename T=float>
void savetxt(std::string filename, std::vector<T> &field)
{
    std::ofstream myfile;
    myfile.open(filename);
    for(auto &i:field)
    {
        myfile << i << '\n';
    }
    myfile.close();
}

/// @brief Usage: Timer t("timer_name");
///               t.start();
///               //do something
///               t.end();
class Timer
{
public:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
    std::chrono::time_point<std::chrono::steady_clock> m_end;
    std::chrono::duration<double, std::milli> elapsed_ms;
    std::chrono::duration<double> elapsed_s;
    std::string name = "";

    Timer(std::string name = "") : name(name){};
    inline void start()
    {
        m_start = std::chrono::steady_clock::now();
    };
    inline void end(string msg = "", string unit = "ms", bool verbose=true, string endsep = "\n")
    {
        m_end = std::chrono::steady_clock::now();
        if (unit == "s")
        {
            elapsed_s = m_end - m_start;
            if(verbose)
                printf("%s(%s): %.0f(s)", msg.c_str(), name.c_str(), elapsed_s.count());
            else
                printf("%.0f(s)", elapsed_s.count());
        }
        else //else if(unit == "ms")
        {
            elapsed_ms = m_end - m_start;
            if(verbose)
                printf("%s(%s): %.0f(ms)", msg.c_str(), name.c_str(), elapsed_ms.count());
            else
                printf("%.0f(ms)", elapsed_ms.count());
        }
        printf("%s", endsep.c_str());
    }
    inline void reset()
    {
        m_start = std::chrono::steady_clock::now();
        m_end = std::chrono::steady_clock::now();
    };
};
Timer global_timer("global");

// caution: the tic toc cannot be nested
inline void tic()
{
    global_timer.reset();
    global_timer.start();
}

inline void toc(string message = "")
{
    global_timer.end(message);
    global_timer.reset();
}


extern "C" DLLEXPORT int spgemm(int* indptr, int* indices, float* data, int nrows, int ncols, int nnz,
int* indptr2, int* indices2, float* data2, int nrows2, int ncols2, int nnz2)
{
    // Host problem definition
    int A_num_rows = nrows;
    int A_num_cols = ncols;
    int A_nnz      = nnz;
    int B_num_rows = nrows2;
    int B_num_cols = ncols2;
    int B_nnz      = nnz2;

    float               alpha       = 1.0f;
    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    tic();
    // Device memory management: Allocate and copy A, B
    int   *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
          *dC_csrOffsets, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    // allocate A
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )
    // allocate B
    CHECK_CUDA( cudaMalloc((void**) &dB_csrOffsets,
                           (B_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dB_values,  B_nnz * sizeof(float)) )
    // allocate C offsets
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )

    // copy A
    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, indptr,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, indices, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, data,
                           A_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    // copy B
    CHECK_CUDA( cudaMemcpy(dB_csrOffsets, indptr2,
                           (B_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_columns, indices2, B_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_values, data2,
                           B_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                      dB_csrOffsets, dB_columns, dB_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                      dC_csrOffsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    toc("memcpy and create descriptors");
    tic();
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )
    toc("spgemm computation");
    tic();

    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )
    // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )


    //--------------------------------------------------------------------------
    // // device result check
    std::vector<int> hC_csrOffsets_tmp(C_num_rows1 + 1);
    std::vector<int> hC_columns_tmp(C_nnz1);
    std::vector<float> hC_values_tmp(C_nnz1);

    CHECK_CUDA( cudaMemcpy(hC_csrOffsets_tmp.data(), dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_columns_tmp.data(), dC_columns, C_nnz1 * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_values_tmp.data(), dC_values, C_nnz1 * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    toc("memcpy back");
    tic();
    std::cout << "spgemm_example test PASSED" << std::endl;
    std::cout << "C_nnz: " << C_nnz1 << std::endl;
    std::cout << "save C in txt" << std::endl;
    savetxt("C.data.txt", hC_values_tmp);
    savetxt<int>("C.indptr.txt", hC_csrOffsets_tmp);
    savetxt<int>("C.indices.txt", hC_columns_tmp);
    // for(int i = 0; i < C_nnz1; i++) {
    //     std::cout << hC_values_tmp[i] << " ";
    // }
    std::cout << "save C done" << std::endl;
    toc("save C");
    //--------------------------------------------------------------------------
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB_csrOffsets) )
    CHECK_CUDA( cudaFree(dB_columns) )
    CHECK_CUDA( cudaFree(dB_values) )
    CHECK_CUDA( cudaFree(dC_csrOffsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
    return EXIT_SUCCESS;
}


extern "C" DLLEXPORT void change_spmat(int* indptr, int* indices, double* data, int nrows, int ncols, int nnz)
{
    for (int i=0; i<nrows; i++) 
        for (int j=indptr[i]; j<indptr[i+1]; j++)
            data[j] += 1;
}
