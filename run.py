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

# case1: AMG restart from 200-400  dt=1e-3 N1024
def case1():
    args = ["python3", "engine/cloth/cloth3d.py",
          "-N=64",
          "-solver_type=AMG", 
          "-delta_t=1e-3", 
          "-restart=0", 
          "-restart_frame=200",
          "-end_frame=400"]
    call(args)


if __name__=='__main__':
    print(f'Running case {case_num}...')
    if case_num==1:
        case1()
    elif case_num==2:
        ...
    else:
        print('Invalid case number. Exiting...')
        sys.exit(1)