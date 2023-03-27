import taichi as ti

@ti.data_oriented
class MetaData:
    def __init__(self):
        self.dt = 0.001  # timestep size
        self.inv_h2 = 1.0 / self.dt / self.dt
        self.omega = 0.2
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.MaxIte = 2
        self.numSubsteps = 10

meta = MetaData()