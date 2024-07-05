import sys
sys.path.append('./build/Release')  # 添加模块搜索路径
sys.path.append('./build/Debug')  # 添加模块搜索路径


import rigid_body_modes
import numpy as np
ndim = 2
coo = [0.0, 1.0, 2.0, 3.0]  # 示例坐标向量
coo = np.array(coo, dtype=np.float64)  # 转换为numpy数组
B = np.zeros(shape=coo.shape)  # 输出矩阵将填充到这个列表中
transpose = False  # 可选参数

# 调用函数
B = rigid_body_modes.rigid_body_modes(ndim, coo, B, transpose)
print(B)  # 输出结果