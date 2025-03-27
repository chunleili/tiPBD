# Usage:
# python script/draw/draw_res.py --data result/case216-0327-bunny/latest.log script/draw/214.log --labels "AMG" "XPBD" --title "自定义标题"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='*',  help='数据文件路径列表', default=[])
    parser.add_argument('--labels', nargs='*',  help='每个数据对应的标签')
    parser.add_argument('--title', type=str, default='', help='图表标题')
    parser.add_argument('--colors', nargs='*', default=['b-', 'r-', 'g-', 'k-'], help='线条颜色和样式')
    return parser.parse_args()


def process_data(filename):
    # %%
    # 从txt文件中读取数据
    print(filename)
    def read_txt(filename):
        with open(filename, 'r') as f:
            raw = f.readlines()
        return raw

    raw = read_txt(filename)
    # print(raw)

    # %%
    # 解析数据
    data = []
    for line in raw:
        try:
            if line.startswith("Frame"):
                parts = line.strip().split(' ')
                # Skip lines that don't have enough parts
                if len(parts) < 5:
                    continue
                    
                # Extract only valid data entries
                valid_parts = [p for p in parts if ':' in p]
                if len(valid_parts) < 5:
                    continue

                frame = int(valid_parts[0].split(':')[1])
                iter_ = int(valid_parts[1].split(':')[1])
                residual = float(valid_parts[2].split(':')[1])
                relative = float(valid_parts[3].split(':')[1])
                frame_past_time = float(valid_parts[4].split(':')[1].replace('ms', ''))
                data.append([frame, iter_, residual, relative, frame_past_time])
        except (IndexError, ValueError) as e:
            print(f"Skipping invalid line: {line.strip()}")
            continue

    df = pd.DataFrame(data, columns=['Frame', 'Iter', 'Residual', 'Relative', 'FramePastTime'])

    # 将FramePastTime从毫秒转换为秒
    df['FramePastTime'] = df['FramePastTime'] / 1000.0

    return df

    
def plot_comparison(data_files, labels, colors, title=''):
    # 处理数据
    dfs = []
    for file in data_files:
        df = process_data(file)
        df_frame1 = df[df['Frame'] == 1]
        dfs.append(df_frame1)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 设置总标题
    fig.suptitle(title, fontsize=12)

    # 绘制图表
    for df, label, color in zip(dfs, labels, colors):
        ax1.plot(df['Iter'] + 1, df['Relative'], color, label=label)
        ax2.plot(df['FramePastTime'], df['Relative'], color, label=label)

    # 设置坐标轴
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Relative Residual')
    ax1.grid(True)
    ax1.legend()

    ax2.set_yscale('log')
    ax2.set_xlabel('Computation Time (s)')
    ax2.set_ylabel('Relative Residual')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    global thisDir
    thisDir = str(Path(__file__).parent)+'/'
    args = parse_args()
    
    # Convert space-separated string to list
    if isinstance(args.data, str):
        args.data = args.data.split()
    if isinstance(args.labels, str):
        args.labels = args.labels.split()
    
    # Default test data if no arguments provided
    if not args.data:
        args.data = ["result/case216-0327-bunny/latest.log", f"{thisDir}214.log"]
        args.labels = ["AMG", "XPBD"]
    
    plot_comparison(args.data, args.labels, args.colors, args.title)