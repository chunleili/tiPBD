# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

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



config="twist_bar"
fig,axs = plt.subplots(1,squeeze=True)
# df_ruan = process_data(f'energy_{config}_ruan.txt')
df_xian = process_data(f'energy_{config}_xian.txt')
df_amg = process_data(f'energy_{config}_amg.txt')
df_xpbd = process_data(f'energy_{config}_xpbd.txt')
df_direct = process_data(f'energy_{config}_direct.txt')
# axs.plot(df_ruan['Iter'], df_ruan['Value'],label='ruan')
axs.plot(df_xian['Iter'], df_xian['Value'],label='xian')
axs.plot(df_amg['Iter'], df_amg['Value'],label='amg')
axs.plot(df_xpbd['Iter'], df_xpbd['Value'],label='xpbd')
axs.plot(df_direct['Iter'], df_direct['Value'],label='direct',linestyle='-.')
plt.yscale('log')
plt.title(f'{config}')
plt.legend()
plt.xlabel('Iter')
plt.ylabel('Energy')
plt.show()