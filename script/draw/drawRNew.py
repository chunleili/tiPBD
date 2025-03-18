import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def read_time_from_log(log_file):
    with open(log_file, "r") as f:
        # 读取所有字符串
        lines = f.readlines()
        #去掉 leading and trailing whitespaces
        lines = [line.strip() for line in lines]
        # 去掉空行
        lines = [line for line in lines if line]


        # 提取出FramePastTime后面的数字
        FramePastTime = [line for line in lines if f"Time budget left:" in line]
        
        # 按空格分割，并取最后一个
        time_value = []
        for i in range(len(FramePastTime)):
            time_value.append( process_frame_past_time(FramePastTime[i]))
        time_value = np.array(time_value)
    return  time_value


def read_r_from_log(log_file, frame):
    with open(log_file, "r") as f:
        # 读取所有字符串
        lines = f.readlines()
        #去掉 leading and trailing whitespaces
        lines = [line.strip() for line in lines]
        # 去掉空行
        lines = [line for line in lines if line]

        # 提取出特定的frame, 例如以20-开头
        lines = [line for line in lines if line.startswith(f"{frame}-")]

        r = [line for line in lines if f"dual:" in line]
        r0 = float(r[0].split(f"dual0:")[1].split()[0])
        for i,l in enumerate(r):
            r[i] = float(l.split(f"dual:")[1].split()[0])
        r.insert(0, r0)
        r = np.array(r)
    return r

def process_frame_past_time(FramePastTime):
    # 按空格分割字符串，并获取最后一个元素
    last_element = FramePastTime.split()[-1]

    # 去掉 'ms'
    time_value = last_element.replace('ms', '')
    time_value = float(time_value)
    return time_value


# def process(log_file, frame):
#     residual= read_r_from_log(log_file,frame)
#     niter = len(residual)-1 #去除r0
#     print("niter:", niter)
#     FramePastTime= read_time_from_log(log_file)
#     FramePastTime= FramePastTime[:niter+1]
#     print(f"residual:{residual[0]}->{residual[-1]}")
#     return FramePastTime, residual


log_file = "D:\MGPBD/bunny-resolution/bunny-resolution/case163-0116-bunny/latest.log"
residual1= read_r_from_log(log_file,19)
niter = len(residual1)-1 #去除r0
print("niter:", niter)
FramePastTime1= read_time_from_log(log_file)
FramePastTime1= FramePastTime1[:niter+1]
print(f"residual:{residual1[0]}->{residual1[-1]}")
# FramePastTime1 = 1e4 - FramePastTime1
np.savetxt("FramePastTime1.txt", FramePastTime1)
np.savetxt("residual1.txt", residual1)

log_file = "D:\MGPBD/bunny-resolution/bunny-resolution/case167-0116-bunny/latest.log"
FramePastTime2,residual2 = process(log_file,19)
# FramePastTime2 = 1e4 - FramePastTime2
np.savetxt("FramePastTime2.txt", FramePastTime2)
np.savetxt("residual2.txt", residual2)

[
8.090000000000000568e+01,
8.079999999999999716e+01,
7.479999999999999716e+01,
6.929999999999999716e+01,
6.400000000000000000e+01,
5.979999999999999716e+01,
5.560000000000000142e+01,
5.250000000000000000e+01,
4.889999999999999858e+01,
4.560000000000000142e+01,
4.260000000000000142e+01,
4.020000000000000284e+01,]

7.426000000000000000e+03
7.812000000000000000e+03
8.201000000000000000e+03
8.590000000000000000e+03
8.995000000000000000e+03
9.388000000000000000e+03
9.779000000000000000e+03
1.020500000000000000e+04
5.394000000000000000e+03
5.776000000000000000e+03
6.166000000000000000e+03
6.558000000000000000e+03

fig,axs = plt.subplots(1,squeeze=True)
axs.plot(FramePastTime1[0:9], residual1[0:9]/(residual2[0]),color='r',label='MGPBD')
axs.plot(FramePastTime2[0:-100], residual2[0:-100]/(residual2[0]),color='b',label='XPBD')
axs.set_xlabel(f"Frame Past Time(ms)")
axs.set_ylabel("dual residual")
axs.set_yscale("log")
plt.show()