import numpy as np



def make_and_clean_dirs(dir):
    import shutil
    from pathlib import Path

    shutil.rmtree(dir, ignore_errors=True)

    Path(dir).mkdir(parents=True, exist_ok=True)
    Path(dir + "/r/").mkdir(parents=True, exist_ok=True)
    Path(dir + "/A/").mkdir(parents=True, exist_ok=True)
    Path(dir + "/state/").mkdir(parents=True, exist_ok=True)
    Path(dir + "/mesh/").mkdir(parents=True, exist_ok=True)



def use_another_outdir(dir):
    import re
    from pathlib import Path
    path = Path(dir)
    if path.exists():
        # 使用正则表达式匹配文件夹名称中的数字后缀
        base_name = path.name
        match = re.search(r'_(\d+)$', base_name)
        if match:
            base_name = base_name[:match.start()]
            i = int(match.group(1)) + 1
        else:
            base_name = base_name
            i = 1

        while True:
            new_name = f"{base_name}_{i}"
            path = path.parent / new_name
            if not path.exists():
                break
            i += 1

    dir = str(path)
    print(f"\nFind another outdir: {dir}\n")
    return dir



def process_dirs(args):
    if args.auto_another_outdir:
        args.out_dir = use_another_outdir(args.out_dir)
    make_and_clean_dirs(args.out_dir)




def parse_json_params(path, vars_to_overwrite):
    import os, json
    if not os.path.exists(path):
        assert False, f"json file {path} not exist!"
    print(f"CAUTION: using json config file {path} to overwrite the command line args!")
    with open(path, "r") as json_file:
        config = json.load(json_file)
    for key, value in config.items():
        if key in vars_to_overwrite:
            if vars_to_overwrite[key] != value:
                print(f"overwriting {key} from {vars_to_overwrite[key]} to {value}")
                vars_to_overwrite[key] = value
        else:
            print(f"json key {key} not exist in vars_to_overwrite!")



def find_last_frame(dir):
    import glob, os
    from pathlib import Path
    # find the last ist.frame number of dir
    files = glob.glob(dir + "/state/*.npz")
    files.sort(key=os.path.getmtime)
    if len(files) == 0:
        return 0
    path = Path(files[-1])
    last_frame = int(path.stem)
    return last_frame

def do_restart(args,ist):
    if args.restart_from_last_frame :
        args.restart_frame =  find_last_frame(args.out_dir)
    if args.restart_frame == 0:
        print("No restart file found.")
    else:
        load_state(args.restart_dir + f"{args.restart_frame:04d}.npz")
        ist.frame = args.restart_frame
        print(f"restart from last ist.frame: {args.restart_frame}")


def save_state(filename, ist):
    state = [ist.frame, ist.pos, ist.vel, ist.old_pos, ist.predict_pos, ist.rest_len]
    for i in range(1, len(state)):
        state[i] = state[i].to_numpy()
    np.savez(filename, *state)
    print(f"Saved ist.frame-{ist.frame} states to '{filename}', {len(state)} variables")

def load_state(filename,ist):
    npzfile = np.load(filename)
    state = [ist.frame, ist.pos, ist.vel, ist.old_pos, ist.predict_pos, ist.rest_len]
    ist.frame = int(npzfile["arr_0"])
    for i in range(1, len(state)):
        state[i].from_numpy(npzfile["arr_" + str(i)])
    print(f"Loaded ist.frame-{ist.frame} states to '{filename}', {len(state)} variables")



def export_A_b(A, b, dir, postfix=f"", binary=True):
    from time import perf_counter
    import scipy

    print(f"Exporting A and b to {dir} with postfix {postfix}")
    tic = perf_counter()
    if binary:
        # https://stackoverflow.com/a/8980156/19253199
        scipy.sparse.save_npz(dir + f"/A_{postfix}.npz", A)
        if b is not None:
            np.save(dir + f"/b_{postfix}.npy", b)
        # A = scipy.sparse.load_npz("A.npz") # load
        # b = np.load("b.npy")
    else:
        scipy.io.mmwrite(dir + f"/A_{postfix}.mtx", A, symmetry='symmetric')
        if b is not None:
            np.savetxt(dir + f"/b_{postfix}.txt", b)
    print(f"    export_A_b time: {perf_counter()-tic:.3f}s")
    