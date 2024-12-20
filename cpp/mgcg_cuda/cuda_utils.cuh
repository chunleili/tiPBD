#pragma once
#include <cuda_runtime.h>

/* -------------------------------------------------------------------------- */
/*                                 erro check                                 */
/* -------------------------------------------------------------------------- */

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
        printf("%s:%d: %s (%d): %s\n", __FILE__, __LINE__,                     \
               cudaGetErrorString(status), status, #func);                     \
        cuerr();                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("%s:%d: %s (%d): %s\n", __FILE__, __LINE__,                     \
               cusparseGetErrorString(status), status, #func);                 \
        cuerr();                                                               \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("%s:%d: %s (%d): %s\n", __FILE__, __LINE__,                     \
               cublasGetStatusString(status), status, #func);                  \
        cuerr();                                                               \
    }                                                                          \
}


// https://github.com/NVIDIA/CUDALibrarySamples/blob/ed19a07b6dd0900b7547b274a6ed9d7c22a6d431/cuSOLVER/utils/cusolver_utils.h#L55
#define CHECK_CUSOLVER(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            cuerr();                                            \
        }                                                                                          \
    } while (0)




#define LAUNCH_CHECK() \
{\
    cudaError_t varCudaError1 = cudaGetLastError();\
    if (varCudaError1 != cudaSuccess)\
    {\
        printf("Failed to launch kernel, error: %s at %s:%d\n", cudaGetErrorString(varCudaError1), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);\
    }\
}\

/* -------------------------------------------------------------------------- */
/*                               end error check                              */
/* -------------------------------------------------------------------------- */


