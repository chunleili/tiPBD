# %%
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os

prj_dir = Path(__file__).parent.parent.parent
os.chdir(prj_dir)

frame = 74
r_type = "strain"
with0 = False

def read_from_log(log_file, frame, r_type, with0=False):
    with open(log_file, "r") as f:
        # 读取所有字符串
        lines = f.readlines()
        #去掉 leading and trailing whitespaces
        lines = [line.strip() for line in lines]
        # 去掉空行
        lines = [line for line in lines if line]

        # # 提取出特定的frame, 例如以20-开头
        # lines = [line for line in lines if line.startswith(f"{frame}-")]
        # # print(lines)

        # 提取出dual_r后面的数字， dual_r在行中间
        r = [line for line in lines if f"{r_type}:" in line]
        r0 = []
        for i,l in enumerate(r):
            r[i] = float(l.split(f"{r_type}:")[1].split()[0])
            if with0:
                r0.append( float(l.split(f"{r_type}0:")[1].split()[0]))
    return r, r0

log_file = "result/case147-1106-AMG-strain/latest.log"
r, r0 = read_from_log(log_file, frame, r_type, with0)

log_file = "result/case148-1106-XPBD-strain/latest.log"
r2, r02 = read_from_log(log_file, frame, r_type, with0)

print(r2)
if with0:
    print("initial:",r02[0])
    r.insert(0, r02[0])

fig,axs = plt.subplots(2)
axs[0].plot(r[:])
axs[1].plot(r2[:])

plt.xlabel("iteration")
axs[0].set_ylabel(f"{r_type}")
axs[1].set_ylabel(f"{r_type}")

axs[0].legend([f"AMG"])
axs[1].legend([f"XPBD"])
# plt.title(f"{case_name} frame {frame}")
# 设定y轴采用科学计数法
# axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# 设定y轴采用log
axs[0].set_yscale("log")
axs[1].set_yscale("log")
# 设定x轴只能用整数
plt.gca().xaxis.get_major_locator().set_params(integer=True)
plt.show()