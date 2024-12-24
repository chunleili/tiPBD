import os
import subprocess
L = os.listdir('.')
for name in L:
    if name.endswith('.exr'):
        print("converting ", name, " to png")
        subprocess.call(["iconvert", name, name[:-4]+".png"])
print("done")