#include <cuda_runtime.h>
#include <chrono>

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


void launch_check()
{
    cudaError_t varCudaError1 = cudaGetLastError();
    if (varCudaError1 != cudaSuccess)
    {
        std::cout << "Failed to launch kernel (error code: " << cudaGetErrorString(varCudaError1) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
}


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


// Generate random number in the range [0, 1)
struct genRandomNumber {
    __host__ __device__
    float operator()(const int n) const {
        thrust::default_random_engine rng(n);
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng);
    }
};



/* -------------------------------------------------------------------------- */
/*                                    timer                                   */
/* -------------------------------------------------------------------------- */

/// @brief Usage: Timer t("timer_name");
///               t.start();
///               //do something
///               t.end();
/// You need to include <chrono> and <string> for this to work
class Timer
{
public:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
    std::chrono::time_point<std::chrono::steady_clock> m_end;
    std::chrono::duration<double, std::milli> elapsed_ms;
    std::chrono::duration<double> elapsed_s;
    double elapsed=0.0;
    std::string name = "";

    Timer(std::string name = "") : name(name){};
    inline void start()
    {
        m_start = std::chrono::steady_clock::now();
    };
    inline void end(std::string msg = "", std::string unit = "ms", bool verbose=true, std::string endsep = "\n")
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
        elapsed = 0.0;
    };
    inline void accumulate()
    {
        m_end = std::chrono::steady_clock::now();
        elapsed += std::chrono::duration<double, std::milli>(m_end - m_start).count();
    };
    inline void report()
    {
        std::cout << name << ": " << elapsed << " ms" << std::endl;
    };
    
};


// https://stackoverflow.com/a/41154786/19253199
// https://github.com/aramadia/udacity-cs344/blob/master/Unit2%20Code%20Snippets/gputimer.h
// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
/// @brief Usage: 
///     GpuTimer  timer;
///     timer.start(); 
///     do something
///     timer.stop(); 
///     float elapsedTime = timer.elapsed(); 
///     printf("Elapsed time : %.2f ms\n" ,elapsedTime);
struct GpuTimer
{
      cudaEvent_t m_start;
      cudaEvent_t m_stop;

      GpuTimer()
      {
            cudaEventCreate(&m_start);
            cudaEventCreate(&m_stop);
      }

      ~GpuTimer()
      {
            cudaEventDestroy(m_start);
            cudaEventDestroy(m_stop);
      }

      void start()
      {
            cudaEventRecord(m_start, 0);
      }

      void stop()
      {
            cudaEventRecord(m_stop, 0);
      }

      float elapsed()
      {
            float elapsed_ms;
            cudaEventSynchronize(m_stop);
            cudaEventElapsedTime(&elapsed_ms, m_start, m_stop);
            return elapsed_ms;
      }
};
