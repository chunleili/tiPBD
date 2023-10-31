'''
mass spring system
'''

import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import taichi as ti
import time

ti.init(arch=ti.cpu)


pos = np.loadtxt("pos.txt")

print(">>Computing P and R...")
t = time.time()


# ----------------------------------- kmans ---------------------------------- #
print("kmeans start")
input = pos.copy()

M = input.shape[0]
new_M = int(M / 100)

N = M
k = new_M
print("M: ", M)
print("new_M: ", new_M)

# run kmeans
input = whiten(input)
print("whiten done")

print("computing kmeans...")
kmeans_centroids, distortion = kmeans(obs=input, k_or_guess=new_M, iter=5)
labels, _ = vq(input, kmeans_centroids)

print("distortion: ", distortion)
print("kmeans done")



# ---------------------------------------------------------------------------- #
#                                display result                                #
# ---------------------------------------------------------------------------- #
# print("centroids:", centroids)
# print("labels: ", labels)

# ax = plt.subplot(111, projection='3d')
# ax.scatter(input[:,0], input[:,1], input[:,2], c='b')
# ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='r' )
# plt.show()


# Create a color map for each cluster
color_map = ["b", "g", "r", "c", "m", "y", "k"]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot each data point with its corresponding color
for cluster_idx in range(k):
    cluster_points = input[labels == cluster_idx]
    cluster_color = np.random.rand(3)
    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        cluster_points[:, 2],
        c=cluster_color,
        label=f"Cluster {cluster_idx}",
    )

# Plot centroids in black
ax.scatter(input[:, 0], input[:, 1], input[:, 2], c="k", marker="x", s=100, label="Centroids")

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
ax.set_title("Clustered Data Points")
if k <= 20:
    ax.legend()

plt.show()
