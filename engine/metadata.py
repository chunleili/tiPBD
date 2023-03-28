import taichi as ti
import os
from ui.filedialog import filedialog
from ui.config_builder import SimConfig
from ui.parse_commandline_args import parse_commandline_args

def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

@singleton
@ti.data_oriented
class MetaData:
    def __init__(self):
        self.root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.result_path = os.path.join(self.root_path, "result")
        print("root_path:", self.root_path)

        self.geometry_file = "data/model/bunny1k2k/bunny1k.node"
        self.dt = 0.001  # timestep size
        self.inv_h2 = 1.0 / self.dt / self.dt
        self.lame_lambda = 1e5
        self.inv_lame_lambda = 1.0/self.lame_lambda
        self.relax_factor = 0.2
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.ground = ti.Vector([0.0, -2.0, 0.0])
        self.max_iter = 2
        self.num_substeps = 10
        self.frame = 0
        self.step = 0
        self.use_multigrid = True
        self.show_coarse, self.show_fine = True, False
        self.compute_energy = True

        # read from json scene file
        self.use_scene_file = True
        self.use_scene_file = parse_commandline_args().use_scene_file
        if self.use_scene_file:
            self.scene_path = filedialog()
            self.scene_name = self.scene_path.split("/")[-1].split(".")[0]

            self.config = SimConfig(scene_file_path=self.scene_path)

            self.solids = self.config.get_solids()
            self.geometry_file = self.solids[0]["geometry_file"] # 第一个solids
            self.dt = self.config.get_cfg("dt")
            self.inv_h2 = 1.0 / self.dt / self.dt
            self.lame_lambda = self.config.get_cfg("lame_lambda")
            self.inv_lame_lambda = 1.0/self.lame_lambda
            self.relax_factor = self.config.get_cfg("relax_factor")
            self.gravity = ti.Vector(self.config.get_cfg("gravity"))
            self.ground = ti.Vector(self.config.get_cfg("ground"))
            self.max_iter = self.config.get_cfg("max_iter")
            self.num_substeps = self.config.get_cfg("num_substeps")
            self.use_multigrid = self.config.get_cfg("use_multigrid")
            self.show_coarse, self.show_fine = self.config.get_cfg("show_coarse"), self.config.get_cfg("show_fine")
            self.compute_energy = self.config.get_cfg("compute_energy")

            print("\n-----------\nRead parameters from scene file: ", self.scene_path)
            print("geometry_file:", self.geometry_file)
            print("dt:", self.dt)
            print("lame_lambda:", self.lame_lambda)
            print("relax_factor:", self.relax_factor)
            print("gravity:", self.gravity)
            print("ground:", self.ground)
            print("max_iter:", self.max_iter)
            print("num_substeps:", self.num_substeps)
            print("use_multigrid:", self.use_multigrid)
            print("show_coarse:", self.show_coarse)
            print("show_fine:", self.show_fine)
            print("compute_energy:", self.compute_energy)
            print("-----------\n")

meta = MetaData()
