import taichi as ti
import os

@ti.data_oriented
class MetaData:
    def __init__(self):
        self.dt = 0.001  # timestep size
        self.inv_h2 = 1.0 / self.dt / self.dt
        self.omega = 0.2
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.MaxIte = 2
        self.numSubsteps = 10
        self.frame = 0
        self.step = 0
        self.use_multigrid = True
        self.root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.result_path = os.path.join(self.root_path, "result")
        print("root_path:", self.root_path)

meta = MetaData()