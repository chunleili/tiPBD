import os,sys,shutil
import subprocess
from subprocess import call
import argparse


case_num = 1
if '-case' in sys.argv:
    i = sys.argv.index('-case')
    case_num = int(sys.argv[i+1])
else:
    print('Usage: python run.py -case N, with N=1 or 2.\n'
          'Default is 1.\n'
          'Input Choice:\n'
          '1:  AMG restart from 200-400  dt=1e-3 N1024\n'
          '2:  ')

# case1: AMG 1024
def case1():
    args = ["python", "engine/cloth/cloth3d.py",
          "-N=1024",
          "-solver_type=AMG", 
          "-delta_t=1e-3", 
          "-end_frame=400",
          "max_iter=50",
          "max_iter_Axb=150"
          "-scale_instead_of_attach=0",
          "export_matrix_intervel=20"]
    call(args)

# case2: GS N1024
def case2():
    args = ["python", "engine/cloth/cloth3d.py",
          "-N=1024",
          "-solver_type=GS", 
          "-delta_t=1se-3", 
          "-end_frame=400",
          "-scale_instead_of_attach=0"]
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
    call(args)

if __name__=='__main__':
    print(f'Running case {case_num}...')
    if case_num==1:
        case1()
    elif case_num==2:
        case2()
    elif case_num==3:
        case3()
    else:
        print('Invalid case number. Exiting...')
        sys.exit(1)