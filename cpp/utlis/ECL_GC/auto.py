import os, shutil, sys
import subprocess
from pathlib import Path

args = sys.argv
if len(args) < 2:
    print("Usage: python auto.py <.ele file path>")
    sys.exit(1)

path = str(Path(args[1]))
this_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
os.chdir(this_dir)
print("current directory: ", this_dir)


print("copy .ele file to the current directory, rename it to 'input.ele'")

shutil.copy(path, os.path.join(this_dir, "input.ele"))

print("run ECL_GC")
exe = os.path.join(this_dir,"ecl-gc-ColorReduction.exe")

if not os.path.exists(exe):
    print("Please compile the ECL_GC first! Compile with: nvcc -O3 ECL-GC-ColorReduction_12.cu -o ecl-gc-ColorReduction")
    sys.exit(1)

ret = subprocess.call([exe, "ele"], cwd=this_dir)
if ret != 0:
    print("ECL_GC failed")
    sys.exit(1)
print(f"copy color.txt to the original directory: {path}")
shutil.copy(os.path.join(this_dir, "color.txt"), os.path.dirname(path))

os.chdir(cwd)
print("done")