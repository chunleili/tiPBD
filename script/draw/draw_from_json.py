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
    fig,axs = plt.subplots(2)

    df2 = pd.read_json(prj_path + f"result/case3-1018-soft85w-niter2-strengh0.1/r/2.json")
    df= pd.read_json(prj_path + f"result/case4-1018-soft85w-XPBD/r/2.json")

    if draw_time:
        axs[0].plot(df['t'],linestyle='-')
    if draw_dual: 
        axs[0].plot(df['dual'],linestyle='-')
        axs[1].plot(df2['dual'],linestyle='-')

    axs[0].set_xlabel('iteration')
    axs[1].set_xlabel('iteration')

    if draw_time:
        axs[0].set_ylabel('time')
        axs[0].legend(['time'])

    if draw_dual:
        axs[0].set_ylabel('dual residual')
        axs[1].set_ylabel('dual residual')
        axs[0].legend(['XPBD'])
        axs[1].legend(['AMG'])
    
    print(df['dual'].head(1))
    print(df['dual'].tail(1))
    print(df2['dual'].head(1))
    print(df2['dual'].tail(1))

    # axs[0].set_yscale('log')
    plt.tight_layout()
    # axs[0].set_title(f'XPBD dual residual')
    # axs[1].set_title(f'AMG dual residual')

    # 设定y轴采用科学计数法
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # 设定x轴只能用整数
    plt.gca().xaxis.get_major_locator().set_params(integer=True)

    plt.show()


# %%
if __name__ == "__main__":
    draw()