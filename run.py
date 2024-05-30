import os,sys,shutil
import subprocess
from subprocess import call
import argparse


# case1: AMG 1024
def case1():
    args = ["python", "engine/cloth/cloth3d.py",
          "-N=1024",
          "-solver_type=AMG", 
          "-delta_t=1e-3", 
          "-end_frame=200",
          "-max_iter=50",
          "-max_iter_Axb=150"
          "-scale_instead_of_attach=0",
          "-export_matrix_intervel=20"]
    log_args(args)
    call(args)

# case2: GS N1024
def case2():
    args = ["python", "engine/cloth/cloth3d.py",
          "-N=1024",
          "-solver_type=GS", 
          "-delta_t=1e-3", 
          "-end_frame=200",
          "-max_iter=50",
          "-max_iter_Axb=150",
          "-scale_instead_of_attach=0"]
    log_args(args)
    call(args)


# case3: XPBD 1024
def case3():
    args = ["python", "engine/cloth/cloth3d.py",
          "-N=1024",
          "-solver_type=XPBD", 
          "-delta_t=1e-3", 
          "-end_frame=200",
          "-max_iter=100",
          "-max_iter_Axb=150",
          "-scale_instead_of_attach=0",
          ]
    log_args(args)
    call(args)

# case4: XPBD 64
def case4():
    args = ["python", "engine/cloth/cloth3d.py",
          "-N=64",
          "-solver_type=XPBD", 
          "-delta_t=1e-3", 
          "-end_frame=200",
          "-max_iter=100",
          "-max_iter_Axb=150",
          "-scale_instead_of_attach=0",
          ]
    log_args(args)
    call(args)


# case5: profile 1024 attach
def case5():
    args = ["python","-m","cProfile", 
            "-o","profile", "engine/cloth/cloth3d.py",
          "-N=1024",
          "-solver_type=XPBD", 
          "-delta_t=1e-3", 
          "-end_frame=5",
          "-max_iter=100",
          "-max_iter_Axb=150",
          "-scale_instead_of_attach=0",
          "-export_matrix=0",
          "-out_dir=result/profile1024",
          ]
    log_args(args)
    call(args)
    # snakeviz profile
    # call(["snakeviz", "profile"])


def log_args(args:list):
    args1 = " ".join(args) # 将ARGS转换为字符串
    print(f"\nArguments:\n{args1}\n")
    with open("last_run.txt", "w") as f:
        f.write(f"{args1}\n")

if __name__=='__main__':
    if '-case' in sys.argv:
        i = sys.argv.index('-case')
        case_num = int(sys.argv[i+1])
    else:
        print('Usage: python run.py -case N, with N=1 or 2, etc.\n')
    
    # print(f'Running case {case_num}...')
    if case_num==1:
        case1()
    elif case_num==2:
        case2()
    elif case_num==3:
        case3()
    elif case_num==4:
        case4()
    elif case_num==5:
        case5()
    else:
        print('Invalid case number. Exiting...')
        sys.exit(1)