from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
import scipy.sparse
from time import perf_counter
from pathlib import Path



def load_A(path):
    print(f"loading data {path}...")
    path = Path(path)
    if  path.suffix == ".npz":
        binary = True
    elif path.suffix == ".mtx":
        binary = False
    else:
        raise FileNotFoundError(f"Not in mtx or npz format")
    tic = perf_counter()
    if binary:
        # https://stackoverflow.com/a/8980156/19253199
        A = scipy.sparse.load_npz(path)
        A = A.astype(np.float64)
        A = A.tocsr()
    else:
        A = scipy.io.mmread(path)
        A = A.tocsr()
        A = A.astype(np.float64)
    print(f"shape: {A.shape}, nnz: {A.nnz}")
    print(f"loading data {path} done in {perf_counter()-tic:.2f}s")
    return A


def draw(to_read_dir="result/sparsityUA/"):
    fig, axs = plt.subplots(3, figsize=(7, 15),
                            layout="constrained", gridspec_kw={"hspace": 0.1})  # 调整子图大小
    As = [None] * 3
    As[0] = load_A(to_read_dir+"A_L0.npz")
    print("A:", As[0].shape)
    print("nnz:", As[0].nnz)
    axs[0].spy(As[0], markersize=1e-1, markevery=500)#L0

    As[1] = load_A(to_read_dir+"A_L1.npz")
    print("As[1]:", As[1].shape)
    print("nnz:", As[1].nnz)
    axs[1].spy(As[1], markersize=1e-1, markevery=5)#L1

    As[2] = load_A(to_read_dir+"A_L2.npz")
    print("As[2]:", As[2].shape)
    print("nnz:", As[2].nnz)
    axs[2].spy(As[2], markersize=1, markevery=1)#L2

    # axs[0].ticklabel_format(style='sci', axis='both', scilimits=(2,0),useMathText=True,useOffset=True)
    # axs[1].ticklabel_format(style='sci', axis='both', scilimits=(0,0),useMathText=True,useOffset=True)
    # axs[2].ticklabel_format(style='sci', axis='both', scilimits=(2,20),useMathText=True,useOffset=True)

    titles = ["Level 0", "Level 1", "Level 2"]
    sparsity = [As[0].nnz/As[0].shape[0]**2, As[1].nnz/As[1].shape[0]**2, As[2].nnz/As[2].shape[0]**2]
    nnz = [As[0].nnz, As[1].nnz, As[2].nnz]
    # titles = [f"{titles[i]}: {sparsity[i]*100:.1f}%" for i in range(3)]
    titles = [f"{titles[i]}: {sparsity[i]:.0e}" for i in range(3)]
    # titles=[f"Level 0: sparsity" ]
    # titles = [f"{titles[i]}: {nnz[i]:.1e} nnz" for i in range(3)]
    for i, ax in enumerate(axs):
        ax.set_title(titles[i], loc="center", fontsize=15)

    for ax in axs:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(13)
        # ax.xaxis.get_offset_text().set_fontsize(15)
        # ax.yaxis.get_offset_text().set_fontsize(15)
        # ax.yaxis.set_offset_position('left')

        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        # 减少 tick 的数量
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        import matplotlib.ticker as ticker
        # ax.xaxis.set_major_locator(ticker.AutoLocator())
        # ax.yaxis.set_major_locator(ticker.AutoLocator())

    # 调整子图之间的间距
    # plt.subplots_adjust(hspace=1)
    # fig.tight_layout()  # 调整子图之间的间距

    # 增加总标题并放置在底部
    # fig.suptitle('Unsmoothed', fontsize=20, y=0.05)

    # # 在子图之间加横线
    # for i in range(len(axs) - 1):
    #     axs[i].axhline(y=axs[i].get_ylim()[1], color='black', linewidth=1)

    # plt.show()

def generate_data_from_sim():
    import subprocess,os
    # go to the root dir of the project

    print("generating data...")
    print("generating data for UA...")
    args =[
        "python",
        "engine/soft/soft3d.py",
    "-end_frame=2",
    "-out_dir=result/sparsityUA",
    "-model_path=data/model/bunny85w/bunny85w.node",
    "-delta_t=3e-3",
    "-solver_type=AMG",
    "-arch=cpu",
    "-maxiter=20",
    "-smoother_niter=3",
    "-build_P_method=UA",]
    subprocess.check_call(args)

    print("generating data for SA...")
    args =[
        "python",
        "engine/soft/soft3d.py",
    "-end_frame=2",
    "-out_dir=result/sparsitySA",
    "-model_path=data/model/bunny85w/bunny85w.node",
    "-delta_t=3e-3",
    "-solver_type=AMG",
    "-arch=cpu",
    "-maxiter=20",
    "-smoother_niter=3",
    "-build_P_method=SA",]
    subprocess.check_call(args)


if __name__ == "__main__":
    generate_data_from_sim()
    draw("result/sparsityUA/A/")
    draw("result/sparsitySA/A/")
    plt.show()
