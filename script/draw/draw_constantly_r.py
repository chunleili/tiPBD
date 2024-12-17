import json
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
import os,sys
import pandas as pd
import time

def read_json_data(frame):
    json_file = f"{frame}.json"
    while not os.path.exists(json_file):
        time.sleep(5)
        print(f"waiting for {json_file}")
    df = pd.read_json(json_file)
    d = df['obj']
    return d



def read_from_log(log_file, frame, r_type, with0=False):
    with open(log_file, "r") as f:
        # 读取所有字符串
        lines = f.readlines()
        #去掉 leading and trailing whitespaces
        lines = [line.strip() for line in lines]
        # 去掉空行
        lines = [line for line in lines if line]

        # 提取出特定的frame, 例如以20-开头
        lines = [line for line in lines if line.startswith(f"{frame}-")]
        # print(lines)

        # 提取出r_type后面的数字， r_type在行中间
        r = [line for line in lines if f"{r_type}:" in line]
        r0 = []
        for i,l in enumerate(r):
            r[i] = float(l.split(f"{r_type}:")[1].split()[0])
            if with0:
                r0.append( float(l.split(f"{r_type}0:")[1].split()[0]))
    if with0:
        return r, r0
    return r
    

def update_plot(data,frame):
    # 清除当前图表
    plt.clf()
    
    # 绘制新的图表
    plt.plot(data, 'x-')
    plt.xlabel('iteration')
    plt.ylabel(f'{r_type}')
    plt.title(f'{r_type} of frame-{frame}')
    plt.grid(True)
    # plt.ylim(1, 1e8)
    # plt.yscale('log')
    plt.pause(1)
    # plt.savefig(f'./plt/{frame}.png')

def mkdir_if_not_exist(path=None):
    from pathlib import Path
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory_path):
        os.makedirs(path)

prj_path = ((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
from pathlib import Path

os.chdir(prj_path)
out_dir = "result/latest"
os.chdir(out_dir)
mkdir_if_not_exist('./plt')
log_file = "latest.log"
r_type = "dual"

# 不断更新图表
f=1
while True:
    # data = read_json_data(f)
    data = read_from_log(log_file, f, r_type, with0=False)
    if len(data)==0 or len(data)==1:
        print(f"jumping over frame-{f}")
        f+=1
        continue
    update_plot(data,f)
    f+=1
    if f>1000:
        break
