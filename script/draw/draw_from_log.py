# %%
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

prj_dir = Path(__file__).parent.parent.parent

parser = argparse.ArgumentParser()

parser.add_argument("-case_name", type=str, default="case6-0921-cloth1024-AMG-PXPBD_v2_8")
parser.add_argument("-frame", type=int, default=21)
parser.add_argument("-type", type=str, default="dual") 
# "Newton" or "dual" or "primal"

case_name = parser.parse_args().case_name
log_file = prj_dir / f"result/{case_name}/latest.log"

frame = parser.parse_args().frame
r_type = parser.parse_args().type
with0 = False

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

    # 提取出dual_r后面的数字， dual_r在行中间
    r = [line for line in lines if f"{r_type}:" in line]
    r0 = []
    for i,l in enumerate(r):
        r[i] = float(l.split(f"{r_type}:")[1].split()[0])
        if with0:
            r0.append( float(l.split(f"{r_type}0:")[1].split()[0]))
    
print(r)
if with0:
    print("initial:",r0[0])
    r.insert(0, r0[0])

r = np.array(r)
plt.plot(r[:])
plt.xlabel("Outer iteration")
plt.ylabel(f"{r_type} Residual")
plt.legend([f"{r_type}"])
plt.title(f"{case_name} frame {frame}")
# 设定y轴采用科学计数法
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# 设定y轴采用log
plt.yscale("log")
# 设定x轴只能用整数
plt.gca().xaxis.get_major_locator().set_params(integer=True)
plt.show()