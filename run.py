"""
This script is used to batch run the cases.
Usage:
In tiPBD folder, run the following command:

Run single case: `python run.py -cases=1`

Run multiple cases: `python run.py -cases=2 4`

Run multiple cases with 200 frames: `python run.py -cases=2 4 -end_frame=200`

You can modify the cases in the script to add more cases.

last_run.txt in result/meta folder will record the last run command, which is useful for reproducing the result.
"""


import os,sys,shutil
import subprocess
from subprocess import call
import argparse
from time import perf_counter
import logging
import datetime
import re
from pathlib import Path

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

if parser.parse_args().overwrite:
    Warning("Overwrite is enabled. All existing files in the same out_dir name will be overwritten.")
    auto_another_outdir = 1
else:
    auto_another_outdir = 0


def parseNumList(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise argparse.ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))


day = datetime.datetime.now().strftime("%m%d")

allargs = [None]

# naming convention: case{case_num}-{date:4 digits}-{object_type:cloth or soft}{resolution}-{solver_type:AMG or XPBD}

# case1: cloth 1024 
args = ["engine/cloth/cloth3d.py",
                f"-end_frame={end_frame}",
                f"-out_dir=result/case1-{day}-cloth1024-AMG",
                f"-auto_another_outdir={auto_another_outdir}",
        ]
allargs.append(args)

# case2: cloth 1024 XPBD
args = ["engine/cloth/cloth3d.py",
        "-solver_type=XPBD",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case2-{day}-cloth1024-XPBD",
        f"-auto_another_outdir={auto_another_outdir}",
        "-arch=gpu",
        ]
allargs.append(args)

# case3: soft 85w
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case3-{day}-soft85w-AMG",
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
        f"-out_dir=result/case4-{day}-soft85w-XPBD",
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
                f"-out_dir=result/case5-{day}-cloth1024-AMG-PXPBD_v1",
                f"-auto_another_outdir={auto_another_outdir}",
                "-use_PXPBD_v1=1",
                f"-restart=1"
        ]
allargs.append(args)


# case6: cloth 1024 use_PXPBD_v2
args = ["engine/cloth/cloth3d.py",
                f"-end_frame={end_frame}",
                f"-out_dir=result/case6-{day}-cloth1024-AMG-PXPBD_v2",
                f"-auto_another_outdir={auto_another_outdir}",
                "-use_PXPBD_v2=1",
                "-restart=1",
                "-maxiter_Axb=300"
        ]
allargs.append(args)


# case7-68: soft 85w AMGX
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
            f"-out_dir=result/case{case_num}-{day}-soft85w-AMGX-{config_name}",
            f"-auto_another_outdir={auto_another_outdir}",
            "-model_path=data/model/bunny85w/bunny85w.node",
            "-maxiter=50",
            "-tol_Axb=1e-8",
            "-rtol=1e-2",
            "-delta_t=3e-3",
            "-solver_type=AMGX"
            ]
    allargs.append(args)


def run_case(case_num:int):
    args = allargs[case_num]
    if parser.parse_args().profile:
        logging.info(f"Running with cProfile. Output to '{case_num}.profile' file. Use 'snakeviz {case_num}.profile' to view the result.")
        args = [pythonExe,"-m","cProfile", "-o", "profile", *args]
    else:
        args = [pythonExe, *args]
    log_args(args)

    # call(args)

    date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    stderr_file = f"result/meta/stderr_{case_num}.txt"
    with open(stderr_file, "w") as f:
        try:
            subprocess.check_call(args, stderr=f)
        # except Exception as e:
        except subprocess.CalledProcessError as e:
            logging.exception(f"Case {case_num} failed with error: {e}\nDate={date}\nSee {stderr_file} for details\n")
            # raise


def log_args(args:list):
    args1 = " ".join(args) # 将ARGS转换为字符串
    print(f"\nArguments:\n{args1}\n")
    # with open("last_run.txt", "w") as f:
    #     f.write(f"{args1}\n")


# python run.py -end_frame=10 -cases 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 | Tee-Object -FilePath "output.log"
if __name__=='__main__':
    Path("result/meta/").mkdir(parents=True, exist_ok=True)
    if os.path.exists(f'result/meta/batch_run.log'):
        os.remove(f'result/meta/batch_run.log')

    logging.basicConfig(level=logging.INFO, format="%(message)s",filename=f'result/meta/batch_run.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    last_run = " ".join(sys.argv)
    logging.info(f"Date:{date}\nCommand:\n{last_run}\n\n")
    with open("result/meta/last_run.txt", "w") as f:
        f.write(f"{last_run}\n")

    cli_args = parser.parse_args()

    if cli_args.cases:
        tic = perf_counter()
        try:
            for case_num in cli_args.cases:
                if 0 < case_num < len(allargs):
                    print(f'Running case {case_num}...')
                    tic1 = perf_counter()
                    try:
                        date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
                        logging.info(f"Running case {case_num}...\nDate={date}\n")
                        run_case(case_num)
                    except Exception as e:
                        logging.exception(f"Caught exception{e} at case {case_num}, Date={date} continue to next case.\n")  
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
            
    
    if cli_args.list:
        print("All cases:")
        for i in range(len(allargs)):
            print(f"case {i}: {allargs[i]}\n")