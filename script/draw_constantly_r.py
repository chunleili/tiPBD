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



def update_plot(data,frame):
    # 清除当前图表
    plt.clf()
    
    # 绘制新的图表
    plt.plot(data)
    plt.xlabel('iteration')
    plt.ylabel('object')
    plt.title(f'object of frame-{frame}')
    plt.grid(True)
    # plt.ylim(1, 1e8)
    # plt.yscale('log')
    plt.pause(1)
    plt.savefig(f'./plt/{frame}.png')

def mkdir_if_not_exist(path=None):
    from pathlib import Path
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory_path):
        os.makedirs(path)

prj_path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-outdir", type=str, default=prj_path+f"/result/latest/r/")

out_dir = parser.parse_args().outdir
os.chdir(out_dir)
mkdir_if_not_exist('./plt')

# 不断更新图表
f=1
while True:
    data = read_json_data(f)
    update_plot(data,f)
    f+=1
    if f>1000:
        break
