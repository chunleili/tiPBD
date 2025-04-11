# %%
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


os.chdir(Path(__file__).parent)

def process_data(filename):
    # %%
    # # 从txt文件中读取数据
    print(filename)
    def read_txt(filename):
        with open (filename, 'r') as f:
            raw = f.readlines()
        return raw

    raw = read_txt(filename)
    # print(raw)

    # %%
    # to pands
    data = []
    for line in raw:
        line = line.strip().split(' ')
        data.append(line)
    df = pd.DataFrame(data)
    df.columns = ['Frame',"Iter", 'Value']


    # %%
    # 数据处理为数字
    # 去掉前缀
    df['Iter'] = df['Iter'].str.replace('Iter:', '')
    df['Iter'] = df['Iter'].astype(float)
    # print(df)

    # 去掉Frame列
    if 'Frame' in df.columns:
        df = df.drop(columns=['Frame'])

    # 去掉Energy:前缀
    df['Value'] = df['Value'].str.replace('Energy:', '')
    df['Value'] = df['Value'].astype(float)


    # # %%
    # # 归一化： (Value[i]-Value[last])/(Value[0]-Value[last])
    # df['Value'] = df['Value'].astype(float)
    # last_value = df['Value'].iloc[-1]
    # first_value = df['Value'].iloc[0]
    # df['Value'] = (df['Value'] - last_value) / (first_value - last_value)
    # print(df)

    return df


args = parse_args()
# Convert space-separated string to list
if isinstance(args.data, str):
    args.data = args.data.split()

thisDir = str(Path(__file__).parent)+'/'
args.data = [f"{thisDir}energy_twist_bar_7e9_amg_v2.txt",
                f"{thisDir}energy_twist_bar_7e9_xian.txt",
                f"{thisDir}energy_twist_bar_7e9_xpbd.txt",]
                
args.labels = ["AMG_v2", "Xian2019", "XPBD"]

fig,axs = plt.subplots(1,squeeze=True)
dfs = []
for filename in args.data:
    df = process_data(filename)
    dfs.append(
        df)


for i,df in enumerate(dfs):
    axs.plot(df['Iter'], df['Value'],label=args.labels[i])


plt.yscale('log')
plt.title('Energy')
plt.legend()
plt.xlabel('Iter')
plt.ylabel('Energy')
plt.show()