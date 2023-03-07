import taichi as ti
import numpy as np
ti.init(arch=ti.cuda)

#%%
n_particles = 3
dim = 2
n_elements = 1
mu = 1
la = 1
E = 25000
dx = 0.1
x = ti.Vector.field(dim, dtype=float, shape=n_particles, needs_grad=True)
total_energy = ti.field(dtype=float, shape=(), needs_grad=True)
restT = ti.Matrix.field(dim, dim, dtype=float, shape=n_elements)
vertices = ti.field(dtype=ti.i32, shape=(n_elements, 3))

@ti.func
def debug_ti(field:ti.template()):
    print("---------------------")
    print("print inside ti.func")
    print("shape: ",field.shape)
    for i in field:
        print(i, field[i])
    print("---------------------")

@ti.kernel
def compute_total_energy():
    for i in range(n_elements):
        currentT = compute_T(i)
        F = currentT @ restT[i].inverse()
        # NeoHookean
        I1 = (F @ F.transpose()).trace()
        J = F.determinant()
        element_energy = 0.5 * mu * (
            I1 - 2) - mu * ti.log(J) + 0.5 * la * ti.log(J)**2
        total_energy[None] += E * element_energy * dx * dx
@ti.func
def compute_T(i):
    a = vertices[i, 0]
    b = vertices[i, 1]
    c = vertices[i, 2]
    ab = x[b] - x[a]
    ac = x[c] - x[a]
    return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])

@ti.kernel
def initialize():
    vertices[0, 0] = 0
    vertices[0, 1] = 1
    vertices[0, 2] = 2

    x[0] = [0.5, 0.5]
    x[1] = [0.6, 0.5]
    x[2] = [0.5, 0.6]

    for i in range(n_elements):
        restT[i] = compute_T(i)  # Compute rest T


def debug(field):
    field_np = field.to_numpy()
    print("---------------------")
    print("shape: ",field_np.shape)
    print("min, max: ", field_np.min(), field_np.max())
    print(field_np)
    print("---------------------")
    np.savetxt("debug.txt", field_np, fmt="%.4f", delimiter="\t")
    return field_np

initialize()
with ti.ad.Tape(loss=total_energy):
    compute_total_energy()

grad = debug(x.grad)
x_np = debug(x)


#%%
x = ti.field(dtype=ti.f32, shape=(5), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


for i in range(5):
    x[i] = i * 0.1

@ti.kernel
def compute_y():
    for i in range(5):
        y[None] += ti.sin(x[i])

with ti.ad.Tape(y):
    compute_y()

for i in range(5):
    print('dy/dx =', x.grad[i], ' at x =', x[i])

for i in range(5):
    print('cos(x) =', ti.cos(x[i]), ' at x =', x[i])    