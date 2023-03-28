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
        self.frame = 0
        self.step = 0
        self.paused = False

        # read from json scene file
        self.use_scene_file = parse_commandline_args().use_scene_file
        if self.use_scene_file:
            self.scene_path = filedialog()
            self.scene_name = self.scene_path.split("/")[-1].split(".")[0]

            self.config = SimConfig(scene_file_path=self.scene_path)

            self.common = self.config.get_common() # it is a dict
            self.dt = self.common["dt"]
            self.inv_h2 = 1.0 / self.dt / self.dt
            self.relax_factor = self.common["relax_factor"]
            self.gravity = ti.Vector(self.common["gravity"])
            self.ground = ti.Vector(self.common["ground"])
            self.max_iter = self.common["max_iter"]
            self.num_substeps = self.common["num_substeps"]
            self.use_multigrid = self.common["use_multigrid"]
            self.show_coarse, self.show_fine = self.common["show_coarse"], self.common["show_fine"]
            self.compute_energy = self.common["compute_energy"]
            self.use_log = self.common["use_log"]
            self.paused = self.common["initial_pause"]

            self.solids = self.config.get_solids() # it is a list of dict
            self.solid_name = self.solids[0]["name"]
            self.geometry_file = self.solids[0]["geometry_file"] # only support one solid for now
            if self.common['constitutive_model'] == 'arap':
                self.lame_lambda = self.solids[0]["lame_lambda"]
                self.inv_lame_lambda = 1.0/self.lame_lambda
            if self.common['constitutive_model'] == 'neohooken':
                self.youngs_modulus = self.solids[0]["youngs_modulus"]
                self.poisson_ratio = self.solids[0]["poisson_ratio"]
                self.mass_density = self.solids[0]["mass_density"]
                self.lame_lambda = self.youngs_modulus * self.poisson_ratio / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
                self.lame_mu = self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
                self.inv_lame_lambda = 1.0/self.lame_lambda
                self.inv_lame_mu = 1.0/self.lame_mu
                self.gamma = 1 + self.inv_lame_lambda / self.inv_lame_mu # stable neo-hookean


            print("\n-----------\nRead parameters from scene file: ", self.scene_path)
            print("dt:", self.dt)
            print("relax_factor:", self.relax_factor)
            print("gravity:", self.gravity)
            print("ground:", self.ground)
            print("max_iter:", self.max_iter)
            print("num_substeps:", self.num_substeps)
            print("use_multigrid:", self.use_multigrid)
            print("show_coarse:", self.show_coarse)
            print("show_fine:", self.show_fine)
            print("compute_energy:", self.compute_energy)
            print("geometry_file:", self.geometry_file)
            print("-----------\n")

meta = MetaData()
