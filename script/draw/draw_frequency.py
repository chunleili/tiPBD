import matplotlib.pyplot as plt
import scipy
import numpy as np


def draw_frequency_before(r_before, ax, case=""):
    frequencies, spectrum = scipy.signal.periodogram(r_before, return_onesided=False)
    ax.plot(frequencies, spectrum, color="red")
    ax.set_xlabel('Frequency (Hz)', fontsize=15)  # 增大坐标轴字体
    ax.set_ylim([0, ylimit])
    ax.set_title(f'Before', fontsize=15)  # 增大坐标轴字体

def draw_frequency_after(r_after, ax, case=""):
    frequencies, spectrum = scipy.signal.periodogram(r_after, return_onesided=False)
    ax.plot(frequencies, spectrum, color="green")
    ax.set_xlabel('Frequency (Hz)', fontsize=15)  # 增大坐标轴字体
    ax.set_ylim([0, ylimit])
    if case=="XPBD":
        maxiter=maxiterXPBD
    elif case=="MGPBD":
        maxiter=maxiterMGPBD
    ax.set_title(f'{case}(After {maxiter} iters)', fontsize=15)  # 增大坐标轴字体



def draw_fft(signal):
    """
    A hand made version of frequency graph useing numpy.fft.fft
    It generate the same results as scipy.signal.periodogram(signal, return_onesided=False)
    https://stackoverflow.com/a/66845448/19253199
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.fftpack
    fig = plt.figure()
    fourier = np.fft.fft(signal)
    n = signal.size
    timestep = 0.1
    freq = np.fft.fftfreq(n, d=timestep)
    plt.plot(freq, np.abs(fourier))


def generate_data_from_sim():
    import subprocess,os
    # go to the root dir of the project

    print("generating data...")

    print(f"generating frame {frame:04d}.npz from XPBD...")
    args =[
        "python",
        "engine/cloth/cloth3d.py",
        "-out_dir=result/frequencyXPBD",
        "-N=1024",
        f"-end_frame={frame}",
        "-solver_type=XPBD",
        "-maxiter=500",
        "-clean_dir=0",
        "-export_state=1",
        "-calc_dual=0",
        "-export_matrix=1",
    ]
    subprocess.check_call(args)

    print("generating data for XPBD...")
    args =[
        "python",
        "engine/cloth/cloth3d.py",
        "-out_dir=result/frequencyXPBD",
        "-N=1024",
        f"-end_frame={frame}",
        "-solver_type=XPBD",
        f"-maxiter={maxiterXPBD}",
        "-delta_t=3e-3",
        "-clean_dir=0",
        "-export_fulldual=1",
        "-restart=1",
        "-compliance=1e-8",
        f"-restart_file=./result/frequencyXPBD/state/{frame:04d}.npz",
        "-export_matrix=1",
    ]
    subprocess.check_call(args)

    print("generating data for MGPBD...")
    args =[
        "python",
        "engine/cloth/cloth3d.py",
        "-out_dir=result/frequencyMGPBD",
        f"-end_frame={frame}",
        "-N=1024",
        "-solver_type=AMG",
        f"-maxiter={maxiterMGPBD}",
        "-clean_dir=0",
        "-export_fulldual=1",
        "-restart=1",
        f"-restart_file=./result/frequencyXPBD/state/{frame:04d}.npz",
        "-delta_t=3e-3",
        "-compliance=1e-8",
    ]
    subprocess.check_call(args)
    
def load_data(case="XPBD"):
    from pathlib import Path
    import numpy as np
    print("loading...")
    path = Path(f'result/frequency{case}/r/fulldual-{frame}-{0}.npy') 
    r_before = np.load(path)

    if case=="XPBD":
        lastiter=maxiterXPBD-1
    elif case=="MGPBD":
        lastiter=maxiterMGPBD-1
    path = Path(f'result/frequency{case}/r/fulldual-{frame}-{lastiter}.npy') 
    r_after = np.load(path)
    print("load done")
    return r_before, r_after


frame=10
maxiterXPBD = 300
maxiterMGPBD = 2
ylimit = 1e-9
if __name__ == "__main__":
    generate_data_from_sim()
    r_beforeXPBD, r_afterXPBD = load_data("XPBD")
    r_beforeMGPBD, r_afterMGPBD = load_data("MGPBD")

    print(f"rnorm beforeXPBD: {np.linalg.norm(r_beforeXPBD):.2e}")
    print(f"rnorm afterXPBD: {np.linalg.norm(r_afterXPBD):.2e}")
    print(f"rnorm afterMGPBD: {np.linalg.norm(r_afterMGPBD):.2e}")

    fig, axes = plt.subplots(1,3, figsize=(15,4))
    
    draw_frequency_before(r_beforeXPBD, axes[0])
    draw_frequency_after(r_afterXPBD, axes[1], case="XPBD")
    draw_frequency_after(r_afterMGPBD, axes[2], case="MGPBD")
    
    for ax in axes:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)
        ax.xaxis.get_offset_text().set_fontsize(12)
        ax.yaxis.get_offset_text().set_fontsize(12)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True,useOffset=True)
        # ax.yaxis.set_offset_position('left')
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        # 减少 tick 的数量
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0].set_ylabel('Power Spectral Density', fontsize=15)
    fig.savefig(f"result/power_density.png",dpi=300)
    plt.show()

    # draw_fft(r_beforeXPBD)
    # draw_fft(r_afterXPBD)
    # draw_fft(r_afterMGPBD)
    # plt.show()
