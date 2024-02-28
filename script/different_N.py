import os,sys
import subprocess
from subprocess import call

prj_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
print("prj_dir", prj_dir)

os.chdir(prj_dir)


for N in [2,3,4,5,12,32,64]:
    call(["python3", "engine/cloth/cloth3d.py","-N",f"{N}"])
    call(["python3", "script/reproduce_pyamg.py","-N",f"{N}"])
