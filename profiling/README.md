# 性能分析

## Nsight
以管理员身份打开nvidia nsight compute
左上角菜单Connect
executable  E:/App/miniconda3/envs/py310/python.exe
working directory E:/Dev/tiPBD/
command line argument e:/Dev/tiPBD/engine/cloth/cloth3d.py

下面的输出文件随便填一个 例如E:/Dev/tiPBD/NVprof

## VS
打开x64 Native Tools Command Prompt for VS 2022
cd到项目根目录
cl.exe profiling/perf.cpp
右键选择perf.exe为启动项
启动调试器
右下窗口红色小圆点点击记录CPU性能数据