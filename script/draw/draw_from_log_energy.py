# %%
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os

prj_dir = Path(__file__).parent.parent.parent

# parser = argparse.ArgumentParser()

# parser.add_argument("-case_name", type=str, default="case147-AMG-strain")
# parser.add_argument("-frame", type=int, default=0)
# parser.add_argument("-type", type=str, default="dual") 
# "Newton" or "dual" or "primal"

# case_name = parser.parse_args().case_name
# log_file = prj_dir / f"result/{case_name}/{case_name}.log"

# frame = parser.parse_args().frame
# r_type = parser.parse_args().type

prj_dir = Path(__file__).parent.parent.parent
os.chdir(prj_dir)

frame = 75
r_type = "energy"
with0 = False

def read_from_log(log_file, frame, r_type, with0=False):
    with open(log_file, "r") as f:
        # 读取所有字符串
        lines = f.readlines()
        #去掉 leading and trailing whitespaces
        lines = [line.strip() for line in lines]
        # 去掉空行
        lines = [line for line in lines if line]

        # 提取出特定的frame, 例如以20-开头
        lines = [line for line in lines if line.startswith(f"{frame}-")]
        # print(lines)

        # 提取出r_type后面的数字， r_type在行中间
        r = [line for line in lines if f"{r_type}:" in line]
        r0 = []
        for i,l in enumerate(r):
            r[i] = float(l.split(f"{r_type}:")[1].split()[0])
            if with0:
                r0.append( float(l.split(f"{r_type}0:")[1].split()[0]))
    return r, r0

log_file = "result/case149-1106-AMG-energy/latest.log"
r, r0= read_from_log(log_file, frame, r_type, with0)
for i,r_ in enumerate(r):
    print(f"r {r_:.5e}")


log_file ="result/case150-1106-XPBD-energy/latest.log"
r2, r02 = read_from_log(log_file, frame, r_type, with0)
for i,r_ in enumerate(r2):
    print(f"r2 {r_:.5e}")

# if with0:
#     print("initial:",r02[0])
#     r.insert(0, r02[0])

fig,axs = plt.subplots(2)
axs[0].plot(r[:])
axs[1].plot(r2[:])

plt.xlabel("iteration")
axs[0].set_ylabel(f"{r_type}")
axs[1].set_ylabel(f"{r_type}")

axs[0].legend([f"AMG"])
axs[1].legend([f"XPBD"])
# 设定y轴采用科学计数法
# axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# 设定y轴采用log
axs[0].set_yscale("log")
axs[1].set_yscale("log")
# 设定x轴只能用整数
plt.gca().xaxis.get_major_locator().set_params(integer=True)
plt.show()