import numpy as np
import matplotlib.pyplot as plt


def f(A, B):
    res = 4 - np.cos(A) - np.cos(B) - 2*np.cos(A)*np.cos(B)
    return res

A = np.linspace(-2*np.pi, 2*np.pi, 1000)
B = np.linspace(-2*np.pi, 2*np.pi, 1000)


from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X, Y = np.meshgrid(A, B)
Z = f(X, Y)

print("max Z: ", np.max(Z))
print("min Z: ", np.min(Z))
print("(minZ)/(maxZ): ", np.min(Z)/np.max(Z))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-2.01, 2.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()