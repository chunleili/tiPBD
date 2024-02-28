import os,sys
import subprocess
from subprocess import call

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
print("prj_dir", prj_dir)

os.chdir(prj_dir)

for N in [100,200]:
    call(["python", "engine/cloth/cloth2d.py","-N",f"{N}"])
    call(["python", "misc/script/reproduce_pyamg.py","-N",f"{N}"])
