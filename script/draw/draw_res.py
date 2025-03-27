# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

os.chdir(Path(__file__).parent)

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

# 处理数据
df_amg = process_data('210.log')
df_xpbd = process_data('214.log')

# 筛选Frame=1的数据
df_amg_frame1 = df_amg[df_amg['Frame'] == 1]
df_xpbd_frame1 = df_xpbd[df_xpbd['Frame'] == 1]


# 绘制Iter与Relative的关系图
plt.figure(figsize=(7, 6))
# 将迭代次数加1，这样所有值都大于0（这是由于x轴设置为log的时候，0不能作为底数，如果去掉xscale("log")就不需要加1）
plt.plot(df_amg_frame1['Iter'] + 1, df_amg_frame1['Relative'], 'b-', label='AMG') 
plt.plot(df_xpbd_frame1['Iter'] + 1, df_xpbd_frame1['Relative'], 'r-', label='XPBD')
plt.xscale('log')
plt.yscale('log')
plt.title('Iteration vs Relative (Frame 1)')
plt.xlabel('Iteration')
plt.ylabel('Relative')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 绘制FramePastTime与Relative的关系图
plt.figure(figsize=(7, 6))
plt.plot(df_amg_frame1['FramePastTime'], df_amg_frame1['Relative'], 'b-', label='AMG')
plt.plot(df_xpbd_frame1['FramePastTime'], df_xpbd_frame1['Relative'], 'r-', label='XPBD')
plt.yscale('log')
plt.title('Computation Time vs Relative (Frame 1)')
plt.xlabel('Computation Time (s)')
plt.ylabel('Relative')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()