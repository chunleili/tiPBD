# %%
import json
import matplotlib.pyplot as plt
import os,sys
import pandas as pd
from pathlib import Path

# prj_path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
prj_path = Path(__file__).parent.parent.parent
prj_path = str(prj_path)

# define get_col function
get_col = lambda data, col: [d[col] for d in data]
get_col0 = lambda data: get_col(data, 0)
get_col1 = lambda data: get_col(data, 1)

frame=21
draw_time = False
draw_dual = True

# %% draw dual
def draw():
    fig,ax = plt.subplots(1)

    df= pd.read_json(prj_path + f"/result/latest_5/r/{frame}.json")
    if draw_time:
        ax.plot(df['t'],linestyle='-')
    if draw_dual: 
        ax.plot(df['dual'],linestyle='-')

    ax.set_xlabel('iteration')
    if draw_time:
        ax.set_ylabel('time')
        ax.legend(['time'])

    if draw_dual:
        ax.set_ylabel('dual')
        ax.legend(['dual'])

    # ax.set_yscale('log')
    # plt.tight_layout()
    ax.set_title(f'AMG-XPBD')

    # 设定y轴采用科学计数法
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # 设定x轴只能用整数
    plt.gca().xaxis.get_major_locator().set_params(integer=True)

    plt.show()


# %%
if __name__ == "__main__":
    draw()