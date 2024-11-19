"""
This script is used to batch run the cases.
Usage:
In tiPBD folder, run the following command:

Run single case: `python run.py -case=1`

Run multiple cases: `python run.py -case=2 4`

Run multiple cases with 200 frames: `python run.py -case=2 4 -end_frame=200`

You can modify the cases in the script to add more cases.

last_run_batch.txt and last_run_case.text in result/meta folder will record the last run command, which is useful for reproducing the result.
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
parser.add_argument("-case", type=int, nargs='*',help=f"case numbers(can be multiple)")
parser.add_argument("-end_frame", type=int, default=10, help=f"end frame")
parser.add_argument("-overwrite", action="store_true")
parser.add_argument("-A", type=str, help="export a matrix file for testing")

end_frame = parser.parse_args().end_frame

if parser.parse_args().overwrite:
    Warning("Overwrite is enabled. All existing files in the same out_dir name will be overwritten.")
    auto_another_outdir = 1
else:
    auto_another_outdir = 0


def export_A(cli_args):
        # output a matrix for testing
        if cli_args.A == "soft":
                args = ["engine/soft/soft3d.py",
                        f"-end_frame={end_frame}",
                        f"-out_dir=result/test_A",
                        f"-auto_another_outdir={auto_another_outdir}",
                        "-model_path=data/model/bunny85w/bunny85w.node",
                        "-rtol=1e-2",
                        "-tol=1e-4",
                        "-delta_t=3e-3",
                        "-solver_type=AMG",
                        "-arch=cpu",
                        "-maxiter=100",
                        "-smoother_niter=2",
                        "-build_P_method=strength0.1",
                        "-end_frame=1",
                        "-export_matrix=1",
                        ]
                subprocess.check_call([pythonExe] + args)
        
    

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
casenames= {}

# naming convention: case{case_num}-{date:4 digits}-{object_type:cloth or soft}{resolution}-{solver_type:AMG or XPBD}

# case1: cloth 1024 AMG 3ms
case_num = len(allargs)
casenames[case_num] = "cloth-1024-AMG-3ms"
args = ["engine/cloth/cloth3d.py",
        "-solver_type=AMG",
        f"-end_frame=100",
        f"-out_dir=result/case{case_num}-{day}-{casenames[case_num]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-arch=cpu",
        "-N=64",
        "-maxiter=50",
        "-delta_t=10e-3",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-end_frame=100",
        ]
allargs.append(args)

# case2: cloth 64 XPBD gpu 5ms
args = ["engine/cloth/cloth3d.py",
        "-solver_type=XPBD",
        f"-end_frame=100",
        f"-out_dir=result/case{len(allargs)}-{day}-XPBD",
        f"-auto_another_outdir={auto_another_outdir}",
        "-arch=gpu",
        "-N=1024",
        "-maxiter=10000",
        "-delta_t=3e-3",
        "-tol=1e-4",
        "-end_frame=100",
        ]
allargs.append(args)

# case3(from case108): soft 85w AMG 3ms
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-niter2-strengh0.1",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=100",
        "-smoother_niter=2",
        "-build_P_method=strength0.1",
        "-end_frame=1",
        ]
allargs.append(args)

# case4: soft 85w XPBD
args = ["engine/soft/soft3d.py",
        "-solver_type=XPBD",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case4-{day}-soft85w-XPBD",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol=1e-4",
        "-rtol=1e-9",
        "-arch=gpu",
        "-delta_t=3e-3",
        "-end_frame=1",
        "-maxiter=5000",
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


# case69: soft 85w niter5 UA 3ms
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=5",
        "-build_P_method=UA"
        ]
allargs.append(args)


# case70: soft 85w niter5 adaptive 3ms
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5-adaptive",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=5",
        "-build_P_method=adaptive_SA"
        ]
allargs.append(args)


# case71: soft 85w niter5 nullspace 3ms
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5-adaptive",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=5",
        "-build_P_method=nullspace"
        ]
allargs.append(args)


# case72: soft 85w niter5 UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=3",
        "-build_P_method=UA"
        ]
allargs.append(args)

# ==================niter=1 1ms 3ms 5ms==================
# case73: soft85w niter1 1ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter1-1ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=1e-3",
        "-smoother_niter=1",
        "-build_P_method=UA"
        ]
allargs.append(args)


# case74: soft85w niter1 3ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter1-3ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=1",
        "-build_P_method=UA"
        ]
allargs.append(args)


# case75: soft85w niter1 5ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter1-5ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=5e-3",
        "-smoother_niter=1",
        "-build_P_method=UA"
        ]
allargs.append(args)

# ==================niter=5 1ms 3ms 5ms==================
# case76: soft85w niter5 1ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5-1ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=1e-3",
        "-smoother_niter=5",
        "-build_P_method=UA"
        ]
allargs.append(args)


# case77: soft85w niter5 3ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5-3ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=5",
        "-build_P_method=UA"
        ]
allargs.append(args)


# case78: soft85w niter5 5ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5-5ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=5e-3",
        "-smoother_niter=5",
        "-build_P_method=UA"
        ]
allargs.append(args)


# ==================niter=10 1ms 3ms 5ms==================
# case79: soft85w niter10 1ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter10-1ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=1e-3",
        "-smoother_niter=10",
        "-build_P_method=UA"
        ]
allargs.append(args)


# case80: soft85w niter10 3ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter10-3ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=10",
        "-build_P_method=UA"
        ]
allargs.append(args)


# case81: soft85w niter10 5ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter10-5ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=5e-3",
        "-smoother_niter=10",
        "-build_P_method=UA"
        ]
allargs.append(args)


# =============================================
# =============================================
# =======================SA====================
# =============================================


# ==================niter=1 1ms 3ms 5ms==================
# case82: soft85w niter1 1ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter1-1ms-SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=1e-3",
        "-smoother_niter=1",
        "-build_P_method=SA"
        ]
allargs.append(args)


# case83: soft85w niter1 3ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter1-3ms-SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=1",
        "-build_P_method=SA"
        ]
allargs.append(args)


# case84: soft85w niter1 5ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter1-5ms-SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=5e-3",
        "-smoother_niter=1",
        "-build_P_method=SA"
        ]
allargs.append(args)

# ==================niter=5 1ms 3ms 5ms==================
# case85: soft85w niter5 1ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5-1ms-SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=1e-3",
        "-smoother_niter=5",
        "-build_P_method=SA"
        ]
allargs.append(args)


# case86: soft85w niter5 3ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5-3ms-SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=5",
        "-build_P_method=SA"
        ]
allargs.append(args)


# case87: soft85w niter5 5ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter5-5ms-SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=5e-3",
        "-smoother_niter=5",
        "-build_P_method=SA"
        ]
allargs.append(args)


# ==================niter=10 1ms 3ms 5ms==================
# case88: soft85w niter10 1ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter10-1ms-SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=1e-3",
        "-smoother_niter=10",
        "-build_P_method=SA"
        ]
allargs.append(args)


# case89: soft85w niter10 3ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter10-3ms-SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-smoother_niter=10",
        "-build_P_method=SA"
        ]
allargs.append(args)


# case90: soft85w niter10 5ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter10-5ms-SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=5e-3",
        "-smoother_niter=10",
        "-build_P_method=SA"
        ]
allargs.append(args)


# case91: soft85w XPBD 3ms cpu
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-XPBD-3ms",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-solver_type=XPBD",
        "-arch=cpu",
        "-maxiter=3000"
        ]
allargs.append(args)


# case92: cloth 1024 XPBD cpu 1ms
args = ["engine/cloth/cloth3d.py",
        "-solver_type=XPBD",
        f"-end_frame=100",
        f"-out_dir=result/case{len(allargs)}-{day}-cloth1024-XPBD",
        f"-auto_another_outdir={auto_another_outdir}",
        "-arch=cpu",
        "-N=1024",
        "-maxiter=3000",
        "-delta_t=1e-3",
        "-tol=1e-4"
        ]
allargs.append(args)


# case93: soft85w XPBD 3ms gpu
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-XPBD-3ms-gpu",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=3e-3",
        "-solver_type=XPBD",
        "-arch=gpu",
        "-maxiter=3000"
        ]
allargs.append(args)


# case94: soft85w XPBD 1ms gpu
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-XPBD-1ms-gpu",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=1e-3",
        "-solver_type=XPBD",
        "-arch=gpu",
        "-maxiter=3000"
        ]
allargs.append(args)


# case95: soft85w XPBD 5ms gpu
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-XPBD-5ms-gpu",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol_Axb=1e-8",
        "-rtol=1e-2",
        "-delta_t=5e-3",
        "-solver_type=XPBD",
        "-arch=gpu",
        "-maxiter=3000"
        ]
allargs.append(args)


# case96: soft85w XPBD 5ms gpu
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-XPBD-5ms-gpu",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-delta_t=5e-3",
        "-solver_type=XPBD",
        "-arch=gpu",
        "-maxiter=3000"
        ]
allargs.append(args)


# case97: soft85w AMG niter3 3ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-AMG-niter3-3ms-UA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=UA"
        ]
allargs.append(args)


# case98: soft85w AMG niter3 3ms SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=SA"
        ]
allargs.append(args)


# case99: soft85w AMG niter3 3ms nullspace
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=nullspace"
        ]
allargs.append(args)


# case100: soft85w AMG niter3 3ms algebraic3.0
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=algebraic3.0"
        ]
allargs.append(args)


# case101: soft85w AMG niter3 3ms adaptive_SA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-adaptive_SA",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=adaptive_SA"
        ]
allargs.append(args)


# case102: soft85w AMG niter3 3ms strength0.25
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strength0.25",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=strength0.25"
        ]
allargs.append(args)


# case103: soft85w AMG niter3 3ms strength0.3
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strength0.3",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=strength0.3"
        ]
allargs.append(args)


# case104: soft85w AMG niter3 3ms strength0.4
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strength0.4",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=strength0.4"
        ]
allargs.append(args)


# case105: soft85w AMG niter3 3ms strength0.5
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strength0.5",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=strength0.5"
        ]
allargs.append(args)

# case106: soft85w AMG niter3 1ms UA
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-niter3-1ms",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=1e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=UA"
        ]
allargs.append(args)

# case107: soft85w AMG niter3 3ms strength0.2
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strengh0.2",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=strength0.2"
        ]
allargs.append(args)


# case108: soft85w AMG niter3 3ms strength0.1
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strengh0.1",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=strength0.1"
        ]
allargs.append(args)


# case109: soft85w AMG niter3 3ms affinity4.0
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-affinity4.0",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=affinity4.0"
        ]
allargs.append(args)

# case110: soft85w AMG niter3 3ms CAMG
label="CAMG"
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-{label}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=CAMG"
        ]
allargs.append(args)



# case111: soft85w AMG niter3 3ms strength0.1(on case 108) export matrix for test solver diagnostic
args = ["engine/soft/soft3d.py",
        f"-end_frame=1",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strengh0.1",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=3",
        "-build_P_method=strength0.1",
        "-export_matrix=1",
        "-export_matrix_binary=0"
        ]
allargs.append(args)



# case112: soft85w AMG niter2 3ms strength0.1(on case 108, change niter to 2) 
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-niter2-strengh0.1",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=2",
        "-build_P_method=strength0.1"
        ]
allargs.append(args)



# case113: soft85w AMG niter2 3ms evolution
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-evolution-niter2",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=2",
        "-build_P_method=evolution"
        ]
allargs.append(args)



# case114: soft85w AMG niter2 3ms improve_candidate
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-improve_candidate",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=2",
        "-build_P_method=improve_candidate"
        ]
allargs.append(args)



# case115(from case108): soft 85w AMG 3ms for export matrix
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-export_matrix",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=100",
        "-smoother_niter=2",
        "-build_P_method=strength0.1",
        "-end_frame=1",
        "-export_matrix=1",
        "-use_cuda=0"
        ]
allargs.append(args)


# ======================================= 116-119 4ms 5ms
# case116: soft85w 4ms
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-4ms",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=4e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=50",
        "-smoother_niter=2",
        "-build_P_method=strength0.1"
        ]
allargs.append(args)


# case117: soft85w AMG 5ms
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-5ms",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=5e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=50",
        "-smoother_niter=2",
        "-build_P_method=strength0.1"
        ]
allargs.append(args)


# case118: soft 85w XPBD 4ms
args = ["engine/soft/soft3d.py",
        "-solver_type=XPBD",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-XPBD-4ms",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol=1e-4",
        "-rtol=1e-9",
        "-arch=gpu",
        "-delta_t=4e-3",
        "-maxiter=10000",
        ]
allargs.append(args)



# case119: soft 85w XPBD 5ms
args = ["engine/soft/soft3d.py",
        "-solver_type=XPBD",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-XPBD-5ms",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-tol=1e-4",
        "-rtol=1e-9",
        "-arch=gpu",
        "-delta_t=5e-3",
        "-maxiter=10000",
        ]
allargs.append(args)

# ============================120-122 strength

# case120: soft85w AMG strength_energy
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strengh_energy",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=10000",
        "-smoother_niter=2",
        "-build_P_method=strength_energy"
        ]
allargs.append(args)


# case121: soft85w AMG strength_classical
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strength_classical",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=10000",
        "-smoother_niter=2",
        "-build_P_method=strength_classical"
        ]
allargs.append(args)



# case122: soft85w AMG strength_distance
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-strength_distance",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=10000",
        "-smoother_niter=2",
        "-build_P_method=strength_distance"
        ]
allargs.append(args)

# ===================123-126 aggregate standard/naive/lloyd/pairwise
# case123: soft85w AMG aggregate_standard
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-aggregate_standard",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=10000",
        "-smoother_niter=2",
        "-build_P_method=aggregate_standard"
        ]
allargs.append(args)


# case124: soft85w AMG aggregate_naive
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-aggregate_naive",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=10000",
        "-smoother_niter=2",
        "-build_P_method=aggregate_naive"
        ]
allargs.append(args)


# case125: soft85w AMG aggregate_lloyd
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-aggregate_lloyd",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=10000",
        "-smoother_niter=2",
        "-build_P_method=aggregate_lloyd"
        ]
allargs.append(args)


# case126: soft85w AMG aggregate_pairwise
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-aggregate_pairwise",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=10000",
        "-smoother_niter=2",
        "-build_P_method=aggregate_pairwise"
        ]
allargs.append(args)




# case127: soft85w AMG niter2 3ms strength0.1(on case 112, change strength to 0.25) 
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-niter2-strengh0.25",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=2",
        "-build_P_method=strength0.25"
        ]
allargs.append(args)


# case128: soft85w set 01 P
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-01_P",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=2",
        "-build_P_method=strength0.25",
        "-filter_P=01"
        ]
allargs.append(args)


# case129: soft85w set avg P
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-avg_P",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=2",
        "-build_P_method=strength0.25",
        "-filter_P=avg"
        ]
allargs.append(args)



# case130: soft85w set scale_RAP
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-scale_RAP",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=3000",
        "-smoother_niter=2",
        "-build_P_method=strength0.25",
        "-scale_RAP=1"
        ]
allargs.append(args)



# case131: soft85w gauss seidel smoother
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-gauss_seidel_smoother",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-smoother_type=gauss_seidel",
        "-build_P_method=strength0.1"
        ]
allargs.append(args)


# case132: soft85w  only_smoother
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-only_smoother",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-smoother_type=gauss_seidel",
        "-only_smoother=1",
        "-maxiter_Axb=20",
        "-build_P_method=strength0.1"
        ]
allargs.append(args)


# case133: soft85w 33ms 
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-33ms",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        # "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=33e-3",
        "-solver_type=AMG",
        "-smoother_type=jacobi",
        "-smoother_niter=2",
        "-build_P_method=nullspace",
        "-maxiter_Axb=300",
        "-maxiter=100",
        ]
allargs.append(args)


# case134: soft85w 33ms
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-33ms-XPBD",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        # "-rtol=1e-2",
        "-tol=1e-4",
        "-delta_t=33e-3",
        "-solver_type=XPBD",
        "-maxiter=10000",
        ]
allargs.append(args)



# case135: soft85w 3ms  omega=1.0
casenames[len(allargs)] = "soft85w-omega1.0"
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        # "-rtol=1e-2",
        "-omega=1.0",
        # "-mu=1e8",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-smoother_type=jacobi",
        "-smoother_niter=2",
        "-build_P_method=nullspace",
        "-maxiter_Axb=300",
        "-maxiter=100",
        ]
allargs.append(args)


# case136: soft85w 3ms omega=1.0
casenames[len(allargs)] = "soft85w-XPBD-omega1.0"
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-soft85w-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        # "-rtol=1e-2",
        "-solver_type=XPBD",
        "-maxiter=10000",
        "-arch=gpu",
        "-omega=1.0",
        "-mu=1e8",
        "-tol=1e-4",
        "-delta_t=3e-3",
        ]
allargs.append(args)



# case137: soft85w 3ms  large mu
casenames[len(allargs)] = "soft85w-large-mu"
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-rtol=1e-2",
        "-mu=1e9",
        "-tol=1e-4",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-smoother_type=jacobi",
        "-smoother_niter=2",
        "-build_P_method=strength0.1",
        "-maxiter_Axb=100",
        "-maxiter=100",
        "-tol_Axb=1e-6",
        ]
allargs.append(args)


# case138: soft85w 3ms  large mu XPBD
casenames[len(allargs)] = "soft85w-XPBD-large-mu"
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/bunny85w/bunny85w.node",
        "-solver_type=XPBD",
        "-arch=gpu",
        "-maxiter=10000",
        "-rtol=1e-2",
        "-mu=1e9",
        "-tol=1e-4",
        "-delta_t=3e-3",
        ]
allargs.append(args)




# case139: cloth XPBD extreme
casenames[len(allargs)] = "cloth-XPBD-extreme"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-solver_type=XPBD",
        "-arch=gpu",
        "-maxiter=100",
        "-tol=1e-3",
        "-delta_t=3e-3",
        "-N=64",
        "-compliance=1e-8",
        "-end_frame=1000",
        ]
allargs.append(args)


# case140: cloth AMG extreme
casenames[len(allargs)] = "cloth-AMG-extreme"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-solver_type=AMG",
        "-arch=gpu",
        "-maxiter=100",
        "-tol=1e-3",
        "-delta_t=3e-3",
        "-N=64",
        "-compliance=1e-8",
        "-end_frame=1000",
        ]
allargs.append(args)


# case141: cloth global GS
casenames[len(allargs)] = "cloth-global-GS"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-solver_type=GS",
        "-arch=cpu",
        "-maxiter=50",
        "-maxiter_Axb=100",
        "-tol=1e-3",
        "-delta_t=3e-3",
        "-N=64",
        "-end_frame=200",
        ]
allargs.append(args)

# case142: cloth global AMG
casenames[len(allargs)] = "cloth-global-AMG"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-tol=1e-3",
        "-N=64",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-arch=cpu",
        "-maxiter=50",
        "-maxiter_Axb=20",
        "-smoother_niter=2",
        "-build_P_method=strength0.1",
        "-end_frame=100",
        "-export_matrix=1",
        "-export_matrix_frame=80"

        ]
allargs.append(args)



# case143: cloth global GS ninner=100
casenames[len(allargs)] = "cloth-global-GS"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-solver_type=GS",
        "-arch=cpu",
        "-maxiter=50",
        "-maxiter_Axb=100",
        "-tol=1e-3",
        "-delta_t=3e-3",
        "-N=64",
        "-end_frame=200",
        ]
allargs.append(args)




# case144: cloth global GS ninner=1
casenames[len(allargs)] = "cloth-global-GS-1"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-solver_type=GS",
        "-arch=cpu",
        "-maxiter=50",
        "-maxiter_Axb=1",
        "-tol=1e-5",
        "-tol_Axb=1e-6",
        "-delta_t=3e-3",
        "-N=256",
        "-end_frame=100",
        ]
allargs.append(args)

# case145: cloth global AMG
casenames[len(allargs)] = "cloth-global-AMG"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-tol=1e-3",
        "-N=256",
        "-delta_t=3e-3",
        "-solver_type=AMG",
        "-tol=1e-5",
        "-tol_Axb=1e-6",
        "-arch=cpu",
        "-maxiter=50",
        "-maxiter_Axb=20",
        "-smoother_niter=2",
        "-build_P_method=strength0.1",
        "-end_frame=100",
        ]
allargs.append(args)

# case146: cloth global GS ninner=100
casenames[len(allargs)] = "cloth-global-GS-100"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-solver_type=GS",
        "-arch=cpu",
        "-maxiter=50",
        "-maxiter_Axb=100",
        "-tol=1e-5",
        "-tol_Axb=1e-6",
        "-delta_t=3e-3",
        "-N=256",
        "-end_frame=100",
        ]
allargs.append(args)


# case147: AMG-strain
casenames[len(allargs)] = "AMG-strain"
args = ["engine/cloth/cloth3d.py",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-end_frame=75",
        "-delta_t=3e-3",
        "-calc_strain=1",
        "-export_strain=1",
        "-restart=1",
        "-restart_file=result/meta/0074.npz",
        ]
allargs.append(args)


# case148: XPBD-strain
casenames[len(allargs)] = "XPBD-strain"
args = ["engine/cloth/cloth3d.py",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-end_frame=75",
        "-delta_t=3e-3",
        "-solver_type=XPBD",
        "-arch=gpu",
        "-restart=1",
        "-restart_file=result/meta/0074.npz",
        "-maxiter=10000",
        "-calc_strain=1",
        "-export_strain=1",
        ]
allargs.append(args)


# case149: 
casenames[len(allargs)] = "AMG-energy"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-end_frame=75",
        "-delta_t=3e-3",
        "-calc_energy=1",
        "-restart=1",
        "-restart_file=result/meta/0074.npz",
        ]
allargs.append(args)


# case150: 
casenames[len(allargs)] = "XPBD-energy"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-end_frame=75",
        "-delta_t=3e-3",
        "-solver_type=XPBD",
        "-arch=gpu",
        "-maxiter=10000",
        "-calc_energy=1",
        "-restart=1",
        "-restart_file=result/meta/0074.npz",
        ]
allargs.append(args)

# case151 NEWTON
casenames[len(allargs)] = "XPBD-energy"
args = ["engine/cloth/cloth3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-solver_type=NEWTON",
        "-cloth_mesh_type=txt",
        "-delta_t=0.0333",
        "-use_cuda=1",
        "-maxiter=10",
        "-calc_energy=1",
        "-compliance=0.00833333333333333", #1.0/120.0
        "-N=20"
        ]
allargs.append(args)


# case152: 
casenames[len(allargs)] = "XPBD-energy-soft"
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-small",
        "-mu=1e8",
        "-rtol=1e-3",
        "-delta_t=3e-3",
        "-solver_type=XPBD",
        "-maxiter=10000",
        "-calc_energy=1",
        ]
allargs.append(args)



# case153: 
casenames[len(allargs)] = "XPBD-ball"
args = ["engine/soft/soft3d.py",
        f"-end_frame={end_frame}",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/ball/ball.node",
        "-mu=1e8",
        "-rtol=1e-3",
        "-delta_t=3e-3",
        "-solver_type=XPBD",
        "-maxiter=1000",
        "-calc_energy=1",
        "-reinit=freefall",
        "-end_frame=110",
        ]
allargs.append(args)

# case154: 
casenames[len(allargs)] = "XPBD-ball"
args = ["engine/soft/soft3d.py",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/ball22k/ball22k.node",
        "-mu=1e9",
        "-rtol=1e-2",
        "-tol=1e-3",
        "-delta_t=5e-3",
        "-solver_type=XPBD",
        "-maxiter=50",
        # "-calc_energy=1",
        "-reinit=freefall",
        "-end_frame=330",
        "-use_ground_collision=1"
        ]
allargs.append(args)

# case155: 
casenames[len(allargs)] = "AMG-ball"
args = ["engine/soft/soft3d.py",
        f"-out_dir=result/case{len(allargs)}-{day}-{casenames[len(allargs)]}",
        f"-auto_another_outdir={auto_another_outdir}",
        "-model_path=data/model/ball22k/ball22k.node",
        "-mu=1e9",
        "-rtol=1e-2",
        "-tol=1e-3",
        "-delta_t=5e-3",
        "-solver_type=AMG",
        "-maxiter=50",
        # "-calc_energy=1",
        "-reinit=freefall",
        "-end_frame=330",
        "-use_ground_collision=1"
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
    with open("result/meta/last_run_case.txt", "w") as f:
        f.write(f"{args1}\n")

def get_date():
    return datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")


# python run.py -end_frame=10 -cases  63 64 65 66 67 68 | Tee-Object -FilePath "output.log"
if __name__=='__main__':
    Path("result/meta/").mkdir(parents=True, exist_ok=True)
    if os.path.exists(f'result/meta/batch_run.log'):
        os.remove(f'result/meta/batch_run.log')
    logging.basicConfig(level=logging.INFO, format="%(message)s",filename=f'result/meta/batch_run.log',filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


    date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    last_run = " ".join(sys.argv)
    logging.info(f"Date:{date}\nCommand:\n{last_run}\n\n")
    with open("result/meta/last_run_batch.txt", "w") as f:
        f.write(f"{last_run}\n")

    cli_args = parser.parse_args()


    if cli_args.A is not None:
        export_A(cli_args)

    if cli_args.case:
        tic = perf_counter()
        try:
            for case_num in cli_args.case:
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
            if i in casenames:
                print(f"case {i}: {casenames[i]}")
            print(f"case {i}: {allargs[i]}\n")