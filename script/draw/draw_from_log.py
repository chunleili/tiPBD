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

        # 提取出FramePastTime后面的数字
        FramePastTime = [line for line in lines if f"FramePastTime:" in line]
        for i,l in enumerate(FramePastTime):
            FramePastTime[i] = float(l.split(f"FramePastTime:")[1].split()[0])
    return r, r0, FramePastTime

def run_and_draw(log_file, ax, r0):
    r, r0, FramePastTime= read_from_log(log_file, frame, r_type, with0)
    r = np.array(r)
    r = np.concatenate([r0, r])
    print(r[:5])
    ax.plot(FramePastTime, r)
    ax.set_xlabel(f"Frame Past Time(ms)")
    ax.set_ylabel(f"{r_type}")
    ax.set_yscale("log")
    return r

frame = 2
r_type = "dual"
with0 = True
fig,axs = plt.subplots(1,squeeze=True)
log_file = "result/case166-0116-bunny/latest.log"
r, r0, FramePastTime= read_from_log(log_file, frame, r_type, with0)
r = np.array(r)
print(r[:5])
r0__ = r[0]
axs.plot(FramePastTime, r)
axs.set_xlabel(f"Frame Past Time(ms)")
axs.set_ylabel("dual residual")
axs.set_yscale("log")

log_file = "result/case162-0116-bunny/latest.log"
r, r0, FramePastTime= read_from_log(log_file, frame, r_type, with0)
# run_and_draw(log_file, axs, "MGPBD")
r.insert(0, r0__)
FramePastTime.insert(0, 0)
r = np.array(r)
axs.plot(FramePastTime, r)
axs.legend(["XPBD", "MGPBD"])
# r_type = "Newton"
# run_and_draw(log_file, axs[1], r_type)
# r_type = "energy"
# run_and_draw(log_file, axs[2], r_type)
# r_type = "strain"
# run_and_draw(log_file, axs[3], r_type)
plt.show()

Path("result/case166-0116-bunny/latest.log").mkdir(parents=True, exist_ok=True)