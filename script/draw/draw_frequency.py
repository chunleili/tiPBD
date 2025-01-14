import matplotlib.pyplot as plt
import scipy



def draw_frequency_before(r_before, ax, case=""):
    frequencies, spectrum = scipy.signal.periodogram(r_before, return_onesided=False)
    ax.plot(frequencies, spectrum, color="red")
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(f'Power Spectrum (Before)')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # 去掉 x 坐标轴的 tick

def draw_frequency_after(r_after, ax, case=""):
    frequencies, spectrum = scipy.signal.periodogram(r_after, return_onesided=False)
    ax.plot(frequencies, spectrum, color="green")
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(f'{case} Power Spectrum (After)')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # 去掉 x 坐标轴的 tick

def draw_frequency(r_before, r_after, case=""):
    fig, axes = plt.subplots(1,2)
    draw_frequency_before(r_before, axes[0])
    draw_frequency_after(r_after, axes, case[1])
    fig.savefig(f"result/{case}_frequency.png")


def generate_data_from_sim():
    import subprocess,os
    # go to the root dir of the project

    print("generating data...")

    print("generating frame 0020.npz from XPBD...")
    args =[
        "python",
        "engine/cloth/cloth3d.py",
        "-out_dir=result/frequencyXPBD",
        "-N=256",
        f"-end_frame={frame}",
        "-solver_type=XPBD",
        "-maxiter=500",
        "-clean_dir=0",
        "-export_state=1",
        "-calc_dual=0",
        "-export_matrix=1",
    ]
    # file = f"./result/frequencyXPBD/state/{frame:04d}.npz"
    # if not os.path.exists(file):
    subprocess.check_call(args)

    print("generating data for XPBD...")
    args =[
        "python",
        "engine/cloth/cloth3d.py",
        "-out_dir=result/frequencyXPBD",
        "-N=256",
        f"-end_frame={frame}",
        "-solver_type=XPBD",
        f"-maxiter={maxiterXPBD}",
        "-delta_t=5e-3",
        "-clean_dir=0",
        "-export_fulldual=1",
        "-restart=1",
        "-compliance=1e-9",
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
        "-N=256",
        "-solver_type=AMG",
        f"-maxiter={maxiterMGPBD}",
        "-clean_dir=0",
        "-export_fulldual=1",
        "-restart=1",
        f"-restart_file=./result/frequencyXPBD/state/{frame:04d}.npz",
        "-delta_t=5e-3",
        "-compliance=1e-9",
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
    else:
        lastiter=maxiterMGPBD-1
    path = Path(f'result/frequency{case}/r/fulldual-{frame}-{lastiter}.npy') 
    path = Path(f'result/frequency{case}/r/fulldual-{frame}-{lastiter}.npy') 
    r_after = np.load(path)
    print("load done")
    return r_before, r_after

frame=10
maxiterXPBD = 300
maxiterMGPBD = 10
if __name__ == "__main__":
    generate_data_from_sim()
    r_beforeXPBD, r_afterXPBD = load_data("XPBD")
    r_beforeMGPBD, r_afterMGPBD = load_data("MGPBD")

    # draw_frequency(r_beforeXPBD, r_afterXPBD,"XPBD")
    # draw_frequency(r_before, r_after,"MGPBD")
    fig, axes = plt.subplots(1,3)
    draw_frequency_before(r_beforeXPBD, axes[0])
    draw_frequency_after(r_afterXPBD, axes[1], case="XPBD")
    draw_frequency_after(r_afterMGPBD, axes[2], case="MGPBD")
    fig.savefig(f"result/frequency.png")
    plt.show()