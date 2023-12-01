import subprocess
import time
import os
import shutil

cases = ["cloth3d_256_50_amg",
         ]

for case_i in cases:
    os.chdir(f"result/")
    os.chdir(case_i)

    mp4_file = case_i+".mp4"

    subprocess.call(['powershell', '-Command',"pwd"], shell=True)

    command = ['powershell', '-Command', "img2video", "%04d.png", mp4_file]
    subprocess.call(command, shell=True)
    
    time.sleep(1) 

    # 使用 shutil 模块的 copy2 函数复制文件
    source_file = mp4_file
    destination_file = f"../video/"+mp4_file
    shutil.copy2(source_file, destination_file)
    print(f"copy {source_file} to {destination_file} done.")

    os.chdir("../..")
    print(f"cd to {os.getcwd()} done.")