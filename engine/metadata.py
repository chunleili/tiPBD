import taichi as ti
import numpy as np

@ti.data_oriented
class MetaData:
    def __init__(self):
        # self.cfg = config
        # self.GGUI = GGUI

        # self.domain_start = np.array([0.0, 0.0, 0.0])
        # self.domain_start = np.array(self.cfg.get_cfg("domainStart"))

        # self.domain_end = np.array([1.0, 1.0, 1.0])
        # self.domian_end = np.array(self.cfg.get_cfg("domainEnd"))
        
        # self.domain_size = self.domian_end - self.domain_start

        # self.dim = 3
        # self.dim = self.cfg.get_cfg("dim")


        # self.simulation_method = self.cfg.get_cfg("simulationMethod")

        # # Material
        # self.material_solid = 0
        # self.material_fluid = 1

        # self.particle_num = ti.field(int, shape=())

        self.dt = 0.001  # timestep size
        self.inv_h2 = 1.0 / self.dt / self.dt

meta = MetaData()