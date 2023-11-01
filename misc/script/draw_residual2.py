
import numpy as np
import matplotlib.pyplot as plt


input_data = np.loadtxt("E:/Dev/tiPBD/result/cloth3d_256_subspace_50/residual_30.txt")
input_data = input_data[0:11]

input_data2 = np.loadtxt("E:/Dev/tiPBD/result/cloth3d_50_debug/residual_30.txt")
input_data2 = input_data2[0:11]

plt.figure(figsize=(10, 6))
plt.plot(input_data, label="subspace", marker="x", markersize=10, color="orange")
plt.plot(input_data2, label="fullspace", marker="o", markersize=10, color="blue")
plt.yscale("log")
plt.ylabel("residual(2-norm)")
plt.xlabel("iterations")
plt.legend()
plt.show()
