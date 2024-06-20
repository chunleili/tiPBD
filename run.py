import os,sys,shutil
import subprocess
from subprocess import call
import argparse

pythonExe = "python"
if sys.platform == "darwin":
    pythonExe = "python3"

parser = argparse.ArgumentParser()

parser.add_argument("-pyfile", type=str, default="script/prof_solvers.py", help="python file to run")
parser.add_argument("-case", type=int, default=7,help=f"case number")
parser.add_argument("-list", action="store_true", help="list all cases")

pyfile = parser.parse_args().pyfile
case_num = parser.parse_args().case


allargs = [None]

# case1: AMG 1024
args = [pythonExe, "engine/cloth/cloth3d.py",
        "-N=1024",
        "-solver_type=AMG", 
        "-delta_t=1e-3", 
        "-end_frame=800",
        "-max_iter=100",
        "-max_iter_Axb=100",
        "-setup_num=0",
        "-export_matrix=1",
        "-export_matrix_interval=50"]
allargs.append(args)


# case2: GS N1024
args = [pythonExe, "engine/cloth/cloth3d.py",
        "-N=1024",
        "-solver_type=GS", 
        "-delta_t=1e-3", 
        "-end_frame=200",
        "-max_iter=50",
        "-max_iter_Axb=150",
        "-setup_num=0"]
allargs.append(args)


# case3: XPBD 1024
args = [pythonExe, "engine/cloth/cloth3d.py",
        "-N=1024",
        "-solver_type=XPBD", 
        "-delta_t=1e-3", 
        "-end_frame=200",
        "-max_iter=100",
        "-max_iter_Axb=150",
        "-setup_num=0",
        ]
allargs.append(args)

# case4: XPBD 64
args = [pythonExe, "engine/cloth/cloth3d.py",
        "-N=64",
        "-solver_type=XPBD", 
        "-delta_t=1e-3", 
        "-end_frame=200",
        "-max_iter=100",
        "-max_iter_Axb=150",
        "-setup_num=0",
        ]
allargs.append(args)


# case5: profile 1024 attach amg
args = [pythonExe,"-m","cProfile", 
        "-o","profile", "engine/cloth/cloth3d.py",
        "-N=1024",
        "-solver_type=AMG", 
        "-delta_t=1e-3", 
        "-end_frame=5",
        "-max_iter=100",
        "-max_iter_Axb=150",
        "-setup_num=0",
        "-export_matrix=0",
        "-out_dir=profile1024",
        ]
# snakeviz profile
# call(["snakeviz", "profile"])
allargs.append(args)


# case6: scale 64
args = [pythonExe, "engine/cloth/cloth3d.py",
        "-N=64",
        "-solver_type=AMG", 
        "-delta_t=4e-3", 
        "-end_frame=50",
        "-max_iter=100",
        "-max_iter_Axb=150",
        "-setup_num=1",
        "-export_matrix=1",
        "-export_matrix_interval=1",
        "-out_dir=scale64",
        ]
allargs.append(args)


# case7: attach 64
args = [pythonExe, "engine/cloth/cloth3d.py",
        "-json_path=data/scene/cloth/attach64.json",
        "-use_json=1"
        ]
allargs.append(args)


# case8: use geo vs. no geo_stiffness
args = [pythonExe, "engine/cloth/cloth3d.py",
        "-N=64",
        "-solver_type=AMG", 
        "-delta_t=1e-3", 
        "-end_frame=51",
        "-restart=1",
        "-restart_frame=50",
        "-restart_dir=restart",
        "-max_iter=100",
        "-max_iter_Axb=150",
        "-setup_num=0",
        "-export_matrix=1",
        "-export_matrix_interval=1",
        "-out_dir=geoStiff",
        ]
allargs.append(args)

#case9: profile
pyfileBaseNameNoextention = os.path.splitext(os.path.basename(pyfile))[0]
args = [pythonExe,"-m","cProfile", 
        "-o",f"{pyfileBaseNameNoextention}.prof", pyfile
        ]
allargs.append(args)

#case10: vis profile
args = ["snakeviz", f"{pyfileBaseNameNoextention}.prof"]
allargs.append(args)


def run_case(case_num:int):
    args = allargs[case_num]
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