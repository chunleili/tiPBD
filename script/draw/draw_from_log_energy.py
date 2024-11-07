# %%
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os

prj_dir = Path(__file__).parent.parent.parent
os.chdir(prj_dir)

frame = 74
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


def run_and_draw(log_file, ax):
    r, r0= read_from_log(log_file, frame, r_type, with0)
    r = np.array(r)
    print(r[:5])
    ax.plot(r)
    ax.set_ylabel(f"{r_type}")
    ax.set_yscale("log")
    ax.legend([f"AMG"])

fig,axs = plt.subplots(2)
log_file = "result/case149-1106-AMG-energy/latest.log"
run_and_draw(log_file, axs[0])
log_file = "result/case150-1106-XPBD-energy/latest.log"
run_and_draw(log_file, axs[1])
plt.show()