# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

prj_dir = Path(__file__).parent.parent.parent
os.chdir(prj_dir)


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


def run_and_draw(log_file, ax, legend):
    r, r0= read_from_log(log_file, frame, r_type, with0)
    r = np.array(r)
    print(r[:5])
    ax.plot(r)
    ax.set_ylabel(f"{r_type}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.legend([legend])

frame = 1
r_type = "dual"
with0 = False
fig,axs = plt.subplots(2)
log_file = "result/case140-1118-AMG-energy-soft/case140-1118-AMG-energy-soft.log"

run_and_draw(log_file, axs[0], r_type)
axs[0].xaxis.get_major_locator().set_params(integer=True)
# r_type = "Newton"
# run_and_draw(log_file, axs[1], r_type)
r_type = "energy"
run_and_draw(log_file, axs[1], r_type)
# r_type = "strain"
# run_and_draw(log_file, axs[3], r_type)
axs[1].xaxis.get_major_locator().set_params(integer=True)
# tight layout
plt.tight_layout()
plt.show()