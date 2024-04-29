# %%
import json
import matplotlib.pyplot as plt
import os,sys
import pandas as pd

prj_path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = prj_path+f"./result/test/r/"
os.chdir(out_dir)

# %%
frame = 20
json_file = f"{frame}.json"
df = pd.read_json(json_file)

df

# %%
# 取出'sys'数据
dsys = df['sys']
dsys

# %%
# 取出sys列的第一个元素，组成一个list
dsys0 =[d[0] for d in dsys]
dsys0
# %%
# 同理，取出第二个元素，组成一个list
dsys1 =[d[1] for d in dsys]
dsys1
# %%
# 取出obj数据
dobj = df['obj']
dobj
# %%
# 取出dual数据
ddual = df['dual']
ddual
# %%
# 取出amg数据的第一个元素
damg= df['amg']
damg0 = [d[0] for d in damg]
damg0
# %%
# 取出amg数据的第二个元素
damg1 = [d[1] for d in damg]
damg1
# %%
# 画图，画出sys列的第一个元素
plt.plot(dsys0)
plt.title('sys0')
# %%
# 画出sys列的第二个元素
plt.plot(dsys1)
plt.title('sys1')
plt.legend(['sys1'])
# %%
# 同时画出sys列的第一个和第二个元素
plt.plot(dsys0)
plt.plot(dsys1)
# 增加图例
plt.legend(['sys0', 'sys1'])
# 增加title
plt.title('sys0 and sys1')
# %%
# 画出obj列
plt.plot(dobj)
plt.title('obj of amg')
plt.legend(['obj'])
# %%
# 画出dual列
plt.plot(ddual)
plt.title('dual of amg')
plt.legend(['dual'])
# %%
# 画出amg列的第一个元素和第二个元素
plt.plot(damg0)
plt.plot(damg1)
plt.title('amg residuals')
plt.legend(['amg0', 'amg1'])
# %%
