import subprocess
import time
import os

for i in [1,2,5,10,20,50]:
    os.chdir(f"result/cloth3d_{i}")

    subprocess.call(['powershell', '-Command',"pwd"], shell=True)

    command = ['powershell', '-Command', "img2video", "%04d.png", f"cloth3d_{i}.mp4"]
    subprocess.call(command, shell=True)
    
    time.sleep(3) 

    os.chdir("../..")



