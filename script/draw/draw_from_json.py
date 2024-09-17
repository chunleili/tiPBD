# %%
import json
import matplotlib.pyplot as plt
import os,sys
import pandas as pd

prj_path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = prj_path+f"/result/latest/r/"
# os.chdir(out_dir)

# define get_col function
get_col = lambda data, col: [d[col] for d in data]
get_col0 = lambda data: get_col(data, 0)
get_col1 = lambda data: get_col(data, 1)



# %% draw 4 residuals of frames _
def draw_4_residuals(frames:[1]):
    fig,ax = plt.subplots(2,2)
    ax = ax.flatten()

    df = []
    for i,frame in enumerate(frames):
        json_file = f"{frame}.json"
        df.append(pd.read_json(json_file))

        ax[0].plot(get_col0(df[i]['sys']))
        ax[1].plot(get_col1(df[i]['sys']))
        ax[2].plot(df[i]['obj'])
        ax[3].plot(df[i]['dual'])

        plt.tight_layout()

    ax[0].legend([f'frame-{i}' for i in frames])
    # set marker for each frame
    for i in range(4):
        ax[i].set_yscale('log')
        ax[i].set_xlabel('iteration')
        ax[i].set_ylabel('residual')

    ax[0].set_title(f'sys0')
    ax[1].set_title(f'sys1')
    ax[2].set_title(f'obj')
    ax[3].set_title(f'dual')

    plt.show()


# %% draw primary
def draw_primary_secondary_res(ax):
    # fig,ax = plt.subplots(1,2)

    json_file = f"11.json"
    df = pd.read_json(json_file)

    ax[0].plot(df['primary'], marker='v')
    ax[1].plot(df['dual'], marker='x')

    ax[0].set_title("primary")
    ax[1].set_title("secondary")

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    # plt.show()

# %% draw dual
def draw_dual():
    fig,ax = plt.subplots(1)

    frame=30

    df= pd.read_json(prj_path + f"/result/latest/r/{frame}.json")

    df2= pd.read_json(prj_path + f"/result/xpbd_1/r/{frame}.json")

    ax.plot(df['dual'],linestyle='-')
    ax.plot(df2['dual'],linestyle='--')
    plt.tight_layout()

    ax.legend(['ours'])
    # set marker for each frame
    ax.set_yscale('log')
    ax.set_xlabel('iteration')
    ax.set_ylabel('residual')
    ax.set_title(f'dual residual')

    plt.show()

# %%
if __name__ == "__main__":
    draw_dual()