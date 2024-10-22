import os, shutil, sys
import subprocess
from pathlib import Path

args = sys.argv
if len(args) < 2:
    print("Usage: python auto.py <.ele file path>")
    sys.exit(1)

path = str(Path(args[1]))
this_dir = os.path.dirname(os.path.abspath(__file__))
print("current directory: ", this_dir)
print("copy .ele file to the current directory, rename it to 'input.ele'")

shutil.copy(path, os.path.join(this_dir, "input.ele"))

print("run ECL_GC")
exe = os.path.join(this_dir,"ecl-gc-ColorReduction.exe")
ret = subprocess.call([exe, "ele"], cwd=this_dir)
if ret != 0:
    print("ECL_GC failed")
    sys.exit(1)
print(f"copy color.txt to the original directory: {path}")
shutil.copy(os.path.join(this_dir, "color.txt"), os.path.dirname(path))

print("done")