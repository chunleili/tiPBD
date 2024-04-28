# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

def process_log(log_file):
    with open(log_file, "r") as f:
        # 读取所有字符串
        lines = f.readlines()
        #去掉 leading and trailing whitespaces
        lines = [line.strip() for line in lines]
        # 去掉空行
        lines = [line for line in lines if line]

        # 将行开头为gs的行提取出来
        gs = [line for line in lines if line.startswith("gs")]
        # 提取出gs每一行的r:后面的两个数字，然后存为数组
        gs = [line.split(":")[1].split(",") for line in gs]
        # 按空格分开两个数字
        gs = [line[0].split() for line in gs]
        # 将字符串转为浮点数
        gs = [[float(i) for i in line] for line in gs]
        # 转换为numpy数组
        gs = np.array(gs)

        # 将行开头为amg的行提取出来
        amg = [line for line in lines if line.startswith("amg")]
        # 提取出amg每一行的r:后面的两个数字，然后存为数组
        amg = [line.split(":")[1].split(",") for line in amg]
        # 按空格分开两个数字
        amg = [line[0].split() for line in amg]
        # 将字符串转为浮点数
        amg = [[float(i) for i in line] for line in amg]
        # 转换为numpy数组
        amg = np.array(amg)
        ...

        # 提取出dual_r后面的数字， dual_r在行中间
        dual_r = [line for line in lines if "dual_r" in line]
        dual_r = [line.split("dual_r:")[1].split()[0] for line in dual_r]
        dual_r = [float(i) for i in dual_r]
        dual_r = np.array(dual_r)

        # 提取出object后面的数字， object在行中间
        object = [line for line in lines if "object" in line]
        object = [line.split("object:")[1].split()[0] for line in object]
        object = [float(i) for i in object]
        object = np.array(object)

    # %%
    fig1, ax1 = plt.subplots()
    ax1.plot(amg.flatten(), label="r_amg", marker="o", markersize=5, color="blue")
    ax1.plot(gs.flatten(), label="r_gs", marker="x", markersize=5, color="orange")
    ax1.set_ylabel('residual(2-norm)')
    ax1.set_xlabel("iterations")
    ax1.legend()
    fig1.savefig(f"r.png")
    ax1.set_title(log_file)

    return gs, amg, dual_r, object

gs1, amg1, dual_r1, object1 = process_log("gs.log")
gs2 ,amg2, dual_r2, object2 = process_log("amg.log")



fig2, ax2 = plt.subplots()
ax2.plot(dual_r1, label=f"gs_dualr", marker="o", markersize=5, color="blue")
ax2.plot(dual_r2, label=f"amg_dualr", marker="o", markersize=5, color="orange")
ax2.legend()

fig3, ax3 = plt.subplots()
ax3.plot(object1, label=f"gs_object", marker="o", markersize=5, color="blue")
ax3.plot(object2, label=f"amg_object", marker="o", markersize=5, color="orange")
ax3.legend()

plt.show()

fig2.savefig(f"dualr.png")
fig3.savefig(f"object.png")