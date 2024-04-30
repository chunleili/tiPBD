# %%
import json
import matplotlib.pyplot as plt
import os,sys
import pandas as pd

prj_path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = prj_path+f"./result/latest/r/"
os.chdir(out_dir)


# define get_col function
get_col = lambda data, col: [d[col] for d in data]
get_col0 = lambda data: get_col(data, 0)
get_col1 = lambda data: get_col(data, 1)


frame = 51
json_file = f"{frame}-gs.json"
df_gs = pd.read_json(json_file)

json_file = f"{frame}-amg.json"
df_amg = pd.read_json(json_file)



# %% draw sys0  sys1
fig,ax = plt.subplots(2,2)
ax = ax.flatten()

sys0_amg = get_col0(df_amg['sys'])
sys0_gs = get_col0(df_gs['sys'])
ax[0].plot(sys0_amg, 'r', linewidth=2, linestyle='--')
ax[0].plot(sys0_gs, 'b', linewidth=1)
ax[0].legend(['gs', 'amg'])
ax[0].set_title(f'sys0 of gs and amg frame-{frame}')  

sys1_gs = get_col1(df_gs['sys'])
sys1_amg = get_col1(df_amg['sys'])
ax[1].plot(get_col1(df_gs['sys']), 'r', linewidth=2, linestyle='--')
ax[1].plot(get_col1(df_amg['sys']), 'b', linewidth=1)
ax[1].legend(['gs', 'amg'])
ax[1].set_title(f'sys1 of gs and amg frame-{frame}')  
plt.tight_layout()

ax[2].plot(df_gs['obj'], 'r', linewidth=2, linestyle='--')
ax[2].plot(df_amg['obj'], 'b', linewidth=1)
ax[2].legend(['gs', 'amg'])
ax[2].set_title(f'obj of gs and amg frame-{frame}')

ax[3].plot(df_amg['dual'], 'r', linewidth=2, linestyle='--')
ax[3].plot(df_gs['dual'], 'b', linewidth=1)
ax[3].legend(['amg', 'gs'])
ax[3].set_title(f'dual of gs and amg frame-{frame}')
plt.tight_layout()
