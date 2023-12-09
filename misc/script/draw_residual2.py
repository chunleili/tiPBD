
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tqdm

def draw_once(frame_num=50, type="amg"):
    if type == "amg":
        input_data = np.loadtxt(f"./result/cloth3d_256_50_amg/dual_r_frame_{frame_num}.txt")
    elif type == "xpbd":
        input_data2 = np.loadtxt(f"./result/cloth3d_256_50_xpbd/residual_{frame_num}.txt")

    fig = plt.figure(figsize=(10, 6))
    if type == "amg":
        plt.plot(input_data, label="amg", marker="x", markersize=10, color="orange")
    elif type == "xpbd":
        plt.plot(input_data2, label="xpbd", marker="o", markersize=10, color="blue")
    plt.yscale("log")
    plt.ylabel("residual")
    plt.xlabel("iterations")
    plt.title(f"{type} res256 iter50 frame{frame_num} residual")
    plt.legend()
    # plt.show()

    def mkdir_if_not_exist(path=None):
        directory_path = Path(path)
        directory_path.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(directory_path):
            os.makedirs(path)
    if type == "amg":
        out_dir = f"./result/anim/cloth3d_256_50_amg"
    elif type == "xpbd":
        out_dir = f"./result/anim/cloth3d_256_50_xpbd"
    mkdir_if_not_exist(out_dir)
    fig.savefig(out_dir + f"/residual_{frame_num}.png")

    plt.close(fig)

for i in tqdm.trange(1,1000):
    draw_once(i)
