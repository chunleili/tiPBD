import os,sys,shutil
import subprocess
from subprocess import call
import argparse

pythonExe = "python"
if sys.platform == "darwin":
    pythonExe = "python3"

parser = argparse.ArgumentParser()

parser.add_argument("-case", type=int, default=1,help=f"case number")
parser.add_argument("-list", action="store_true", help="list all cases")
parser.add_argument("-profile", action="store_true", help="profiling")

case_num = parser.parse_args().case

allargs = [None]

# case1: attach 64
args = ["engine/cloth/cloth3d.py",
        "-json_path=data/scene/cloth/attach64.json",
        "-use_json=1"
        ]
allargs.append(args)

# case2: scale 64
args = ["engine/cloth/cloth3d.py",
        "-N=64",
        "-setup_num=1",
        "-out_dir=result/scale64",]
allargs.append(args)

# case3: AMG 1024 
args = ["engine/cloth/cloth3d.py",
        "-json_path=data/scene/cloth/attach1024.json",
        "-use_json=1",
        "-export_matrix=1",
        "-auto_another_outdir=1",]
allargs.append(args)

# case4: soft3dBig Bunny
args = ["engine/soft/soft3d.py",
        "-model_path=data/model/bunnyBig/bunnyBig.node",
        "-solver_type=AMG",
        "-out_dir=result/soft3dBig",
        "-export_matrix=1",
        "-auto_another_outdir=1",
        ]
allargs.append(args)


def run_case(case_num:int):
    args = allargs[case_num]
    if parser.parse_args().profile:
        print("Running with cProfile. Output to 'profile' file. Use 'snakeviz profile' to view the result.")
        args = [pythonExe,"-m","cProfile", "-o", "profile", *args]
    else:
        args = [pythonExe, *args]
    log_args(args)
    call(args)


def log_args(args:list):
    args1 = " ".join(args) # 将ARGS转换为字符串
    print(f"\nArguments:\n{args1}\n")
    with open("last_run.txt", "w") as f:
        f.write(f"{args1}\n")


if __name__=='__main__':

    if parser.parse_args().list:
        print("All cases:")
        for i in range(len(allargs)):
            print(f"case {i}: {allargs[i]}")
        sys.exit(0)

    print(f'Running case {case_num}...')
    
    if 0 < case_num < len(allargs):
        run_case(case_num)
    else:
        print('Invalid case number. Exiting...')
        sys.exit(1)