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
import multiprocessing
import time

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
        f"-out_dir=result/case3-{day}-soft85w-AMG-adaptive",
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


# case69: soft 85w
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case3-{day}-soft85w-AMG-niter5",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-jacobi_niter=5",
        "-build_P_method=UA"
        ]
allargs.append(args)



def run_case(case_num:int):
    if case_num < 1 or case_num >= len(allargs):
        print(f'Invalid case number {case_num}. Exiting...')
        sys.exit(1)
    
    args = allargs[case_num]
    if parser.parse_args().profile:
        logging.info(f"Running with cProfile. Output to '{case_num}.profile' file. Use 'snakeviz {case_num}.profile' to view the result.")
        args = [pythonExe,"-m","cProfile", "-o", "profile", *args]
    else:
        args = [pythonExe, *args]
    log_args(args)

    # call(args)

    # stderr_file = f"result/meta/stderr_{case_num}.txt"
    # with open(stderr_file, "w") as f:
    #     try:
    #         subprocess.check_call(args, stderr=f)
    #     except Exception as e:
    #     # except subprocess.CalledProcessError as e:
    #         date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    #         logging.exception(f"Case {case_num} failed with error\nDate={date}\nSee {stderr_file} for details\n")
    #         # raise
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError as e:
        date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        logging.exception(f"Case {case_num} failed with error\nDate={date}\n\n")
    except KeyboardInterrupt:
        logging.exception(f"KeyboardInterrupt case {case_num}")



def log_args(args:list):
    args1 = " ".join(args) # 将ARGS转换为字符串
    print(f"\nArguments:\n{args1}\n")
    with open("last_run_case.txt", "w") as f:
        f.write(f"{args1}\n")

def get_date():
    return datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")


def get_gpu_mem_usage():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_memory = meminfo.used
    total_memory = meminfo.total
    usage = (used_memory / total_memory)
    pynvml.nvmlShutdown()

    with open("result/meta/gpu_mem_usage.txt", "a") as f:
        f.write(f"{get_date()} usage:{usage*100:.1f}% mem:{used_memory/1024**3:.2f}\n")
    if usage > 0.70:
        logging.exception(f"GPU memory usage is too high: {usage*100:.1f}%")
        raise Exception(f"GPU memory usage is too high: {usage*100:.1f}%")
    return usage


def monitor_gpu_usage(interval=60):
    while True:
        get_gpu_mem_usage()
        time.sleep(interval)


# python run.py -end_frame=10 -cases  63 64 65 66 67 68 | Tee-Object -FilePath "output.log"
if __name__=='__main__':
    Path("result/meta/").mkdir(parents=True, exist_ok=True)
    if os.path.exists(f'result/meta/batch_run.log'):
        os.remove(f'result/meta/batch_run.log')
    logging.basicConfig(level=logging.INFO, format="%(message)s",filename=f'result/meta/batch_run.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # pid = os.getpid()
    # logging.info(f"process PPID: {pid}")
    # # 启动 GPU 使用监控线程
    # monitor_thread = multiprocessing.Process(target=monitor_gpu_usage)
    # monitor_thread.daemon = True
    # monitor_thread.start()

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
                logging.info(f"Running case {case_num}...\nDate={get_date()}\n")
                tic1 = perf_counter()
                run_case(case_num)
                tic2 = perf_counter()
                logging.info(f"\ncase {case_num} finished. Time={(tic2-tic1):.1f}s={(tic2-tic1)/60:.2f} min\n---------\n\n")
            print(f"Batch run time: {(perf_counter()-tic)/60:.2f} min")
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            print(f"At case {case_num}")
            print(f"Batch run time: {(perf_counter()-tic)/60:.2f} min")
        except AssertionError as e:
            print(f"AssertionError: {e}")
            print(f"At case {case_num}")
            print(f"Batch run time: {(perf_counter()-tic)/60:.2f} min")
        finally:
            os._exit(1)
    
    if cli_args.list:
        print("All cases:")
        for i in range(len(allargs)):
            print(f"case {i}: {allargs[i]}\n")