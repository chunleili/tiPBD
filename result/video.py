import subprocess
import time
import os
import shutil

cases = [30,50,100]

for i in cases:
    os.chdir(f"result/cloth3d_{i}")

    subprocess.call(['powershell', '-Command',"pwd"], shell=True)

    command = ['powershell', '-Command', "img2video", "%04d.png", f"cloth3d_{i}.mp4"]
    subprocess.call(command, shell=True)
    
    time.sleep(1) 


    # 使用 shutil 模块的 copy2 函数复制文件
    source_file = f"cloth3d_{i}.mp4"
    destination_file = f"../video/cloth3d_{i}.mp4"
    shutil.copy2(source_file, destination_file)
    print(f"copy {source_file} to {destination_file} done.")

    os.chdir("../..")
    print(f"cd to {os.getcwd()} done.")