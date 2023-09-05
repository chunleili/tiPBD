import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import taichi as ti

ti.init(arch=ti.cpu)

# prepare data
# N = 100
# k = int(N/10)
# input = np.random.rand(N, 3) * 10

# read centroids of tetrahedron
input = np.loadtxt("tet_centroid.txt")
N = input.shape[0]
k = int(N / 100)
print("N: ", N)
print("k: ", k)


# run kmeans
print("kmeans start")

input = whiten(input)
print("whiten done")

centroids, distortion = kmeans(obs=input, k_or_guess=k, iter=20)
labels, _ = vq(input, centroids)


print("kmeans done")
print("distortion: ", distortion)
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
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="k", marker="x", s=100, label="Centroids")

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
ax.set_title("Clustered Data Points")
if k <= 20:
    ax.legend()

plt.show()
