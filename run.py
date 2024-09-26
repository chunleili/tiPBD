"""
This script is used to batch run the cases.
Usage:
In tiPBD folder, run the following command:

Run single case: `python run.py -cases=1`

Run multiple cases: `python run.py -cases=2 4`

Run multiple cases with 200 frames: `python run.py -cases=2 4 -end_frame=200`

You can modify the cases in the script to add more cases.

last_run.txt in tiPBD folder will record the last run command, which is useful for reproducing the result.
"""


import os,sys,shutil
import subprocess
from subprocess import call
import argparse
from time import perf_counter
import logging
import datetime

pythonExe = "python"
if sys.platform == "darwin":
    pythonExe = "python3"

parser = argparse.ArgumentParser()

parser.add_argument("-list", action="store_true", help="list all cases")
parser.add_argument("-profile", action="store_true", help="profiling")
parser.add_argument("-cases", type=int, nargs='*',help=f"case numbers")
parser.add_argument("-end_frame", type=int, default=100, help=f"end frame")
parser.add_argument("-overwrite", action="store_true")

end_frame = parser.parse_args().end_frame

auto_another_outdir = 1

if parser.parse_args().overwrite:
    Warning("Overwrite is enabled. All existing files in the same out_dir name will be overwritten.")
    y = input("Are you sure to overwrite?")




allargs = [None]

# naming convention: case{case_num}-{date:4 digits}-{object_type:cloth or soft}{resolution}-{solver_type:AMG or XPBD}

# case1: cloth 1024 
args = ["engine/cloth/cloth3d.py",
                f"-end_frame={end_frame}",
                "-out_dir=result/case1-0921-cloth1024-AMG",
                f"-auto_another_outdir={auto_another_outdir}",
        ]
allargs.append(args)

# case2: cloth 1024 XPBD
args = ["engine/cloth/cloth3d.py",
        "-solver_type=XPBD",
        f"-end_frame={end_frame}",
        "-out_dir=result/case2-0921-cloth1024-XPBD",
        f"-auto_another_outdir={auto_another_outdir}",
        "-arch=gpu",
        ]
allargs.append(args)

# case3: soft 85w
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        "-out_dir=result/case3-0921-soft85w-AMG",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3"
        ]
allargs.append(args)

# case4: soft 85w XPBD
args = ["engine/soft/soft3d.py",
        "-solver_type=XPBD",
        f"-end_frame={end_frame}",
        "-out_dir=result/case4-0921-soft85w-XPBD",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-arch=gpu",
        "-delta_t=3e-3"
        ]
allargs.append(args)


# case5: cloth 1024 use_PXPBD_v1
args = ["engine/cloth/cloth3d.py",
                f"-end_frame={end_frame}",
                "-out_dir=result/case5-0921-cloth1024-AMG-PXPBD_v1",
                f"-auto_another_outdir={auto_another_outdir}",
                "-use_PXPBD_v1=1",
                f"-restart=1"
        ]
allargs.append(args)


# case6: cloth 1024 use_PXPBD_v2
args = ["engine/cloth/cloth3d.py",
                f"-end_frame={end_frame}",
                "-out_dir=result/case6-0921-cloth1024-AMG-PXPBD_v2",
                f"-auto_another_outdir={auto_another_outdir}",
                "-use_PXPBD_v2=1",
                "-restart=1",
                "-maxiter_Axb=300"
        ]
allargs.append(args)


# case7: soft 85w AMGX
# find all config files in data/config/batch/*.json
for config in os.listdir("data/config/batch"):
    config_name = os.path.basename(config).split(".")[0]
    amgx_config = os.path.join("data/config/batch", config)
    case_num = len(allargs)
    # print(f"config: {config}")
    # print(f"amgx_config: {amgx_config}")
    # print(f"case_num: {len(allargs)}")
    args = ["engine/soft/soft3d.py",
            f"-amgx_config={amgx_config}",
            f"-end_frame={end_frame}",
            f"-out_dir=result/case{case_num}-0921-soft85w-AMGX-{config_name}",
            f"-auto_another_outdir={auto_another_outdir}",
            "-model_path=data/model/bunny85w/bunny85w.node",
            "-tol_Axb=1e-8",
            "-rtol=1e-2",
            "-delta_t=3e-3",
            "-solver_type=AMGX"
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

    # call(args)

    date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    stderr_file = f"result/meta/stderr_{case_num}.txt"
    with open(stderr_file, "w") as stderr_file:
        try:
            subprocess.check_call(args, stderr=stderr_file)
        except subprocess.CalledProcessError as e:
            logging.error(f"Case {case_num} failed with error: {e}\n Date={date} see {stderr_file} for details\n")
            raise


def log_args(args:list):
    args1 = " ".join(args) # 将ARGS转换为字符串
    print(f"\nArguments:\n{args1}\n")
    with open("last_run.txt", "w") as f:
        f.write(f"{args1}\n")


if __name__=='__main__':
    if os.path.exists(f'result/meta/batch_run.log'):
        os.remove(f'result/meta/batch_run.log')

    logging.basicConfig(level=logging.INFO, format="%(message)s",filename=f'result/meta/batch_run.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    logging.info(f"Date:{date}\nCommand:{sys.argv}\n\n")

    if parser.parse_args().cases:
        tic = perf_counter()
        try:
            for case_num in parser.parse_args().cases:
                if 0 < case_num < len(allargs):
                    print(f'Running case {case_num}...')
                    tic1 = perf_counter()
                    try:
                        date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
                        run_case(case_num)
                    except Exception as e:
                        logging.error(f"Caught exception{e} at case {case_num}, Date={date} continue to next case.\n")  
                    tic2 = perf_counter()
                    logging.info(f"\ncase {case_num} finished. Time={(tic2-tic1)/60:.2f} min\n---------\n\n")
                else:
                    print(f'Invalid case number {case_num}. Exiting...')
                    sys.exit(1)
            print(f"Batch run time: {(perf_counter()-tic)/60:.2f} min")
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            print(f"At case {case_num}")
            print(f"Batch run time: {(perf_counter()-tic)/60:.2f} min")
        except AssertionError as e:
            print(f"AssertionError: {e}")
            print(f"At case {case_num}")
            print(f"Batch run time: {(perf_counter()-tic)/60:.2f} min")
            
    
    if parser.parse_args().list:
        print("All cases:")
        for i in range(len(allargs)):
            print(f"case {i}: {allargs[i]}\n")