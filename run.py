"""
This script is used to batch run the cases.
Usage:
In tiPBD folder, run the following command:

Run single case: `python run.py -case=1`

Run multiple cases: `python run.py -multi_cases=2 4`

Run multiple cases with 200 frames: `python run.py -multi_cases=2 4 -end_frame=200`

You can modify the cases in the script to add more cases.

last_run.txt in tiPBD folder will record the last run command, which is useful for reproducing the result.
"""


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
parser.add_argument("-multi_cases", type=int, nargs='*',help=f"multiple cases number")
parser.add_argument("-end_frame", type=int, default=100, help=f"end frame")

case_num = parser.parse_args().case
end_frame = parser.parse_args().end_frame

allargs = [None]

# case1: cloth 1024 
args = ["engine/cloth/cloth3d.py",
                f"-end_frame={end_frame}"
        ]
allargs.append(args)

# case2: soft 85w
args = ["engine/soft/soft3d.py",
                f"-end_frame={end_frame}"
        ]
allargs.append(args)


# case3: cloth 1024 XPBD
args = ["engine/cloth/cloth3d.py",
        "-solver_type=XPBD",
        f"-end_frame={end_frame}"
        ]
allargs.append(args)

# case4: soft 85w XPBD
args = ["engine/soft/soft3d.py",
        "-solver_type=XPBD",
                f"-end_frame={end_frame}"
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
    if parser.parse_args().multi_cases:
        for case_num in parser.parse_args().multi_cases:
            if 0 < case_num < len(allargs):
                run_case(case_num)
            else:
                print(f'Invalid case number {case_num}. Exiting...')
                sys.exit(1)
        sys.exit(0)
    
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