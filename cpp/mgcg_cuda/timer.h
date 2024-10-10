#include <chrono>
#include <string>
#include <iostream>

/// @brief Usage: Timer t("timer_name");
///               t.start();
///               //do something
///               t.end();
///               t.report();
/// You need to include <chrono>, <string> and <iostream> for this to work
class Timer
{
public:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
    std::chrono::time_point<std::chrono::steady_clock> m_end;
    std::chrono::duration<double, std::milli> m_elapsed;
    std::string name = "";

    Timer(std::string name = "") : name(name){};
    
    inline void start()
    {
        m_start = std::chrono::steady_clock::now();
    };
    inline void stop()
    {
        m_end = std::chrono::steady_clock::now();
        m_elapsed = (m_end - m_start);
    }
    inline std::chrono::duration<double, std::milli> elapsed()
    {
        return m_elapsed;
    }
    inline void report()
    {
        printf("(%s): %.0f(ms)", name.c_str(), elapsed.count());
    };
    
};