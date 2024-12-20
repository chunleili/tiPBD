#pragma once

#include <chrono>
#include <string>
#include <iostream>

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
