import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
thisdir = Path(__file__).resolve().parent
print(thisdir/'build/Release')
sys.path.append(str(thisdir/'build/Release'))  # 添加模块搜索路径
sys.path.append(str(thisdir/'build/Debug'))  # 添加模块搜索路径
sys.path.append('E:/Dev/tiPBD/cpp/utlis/build/Release/')  # 添加模块搜索路径
sys.path.append(thisdir)

import pymgpbd  # 确认模块名称是否正确
import numpy as np

model = "E:/Dev/tiPBD/data/model/bunny1k2k/coarse.ele"
result = {}
ncolor = pymgpbd.graph_coloring(model, 5, result)
print(result)  # 输出结果
print(ncolor)  # 输出结果