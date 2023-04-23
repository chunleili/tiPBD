import taichi as ti
import logging

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
        import os
        from ui.filedialog import filedialog
        from ui.config_builder import SimConfig
        from ui.parse_cli import parse_cli

        self.root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.result_path = os.path.join(self.root_path, "result")
        print("root_path:", self.root_path)

        self.args = parse_cli()
        # print("args:", self.args)

        if self.args.scene_file == "":
            self.scene_path = filedialog()
        else:
            self.scene_path = self.args.scene_file
        self.scene_name = self.scene_path.split("/")[-1].split(".")[0]
        self.config_instance = SimConfig(scene_file_path=self.scene_path)
        self.common = self.config_instance.config["common"]
        self.materials = self.config_instance.config["materials"]
        if "sdf_meshes" in self.config_instance.config:
            self.sdf_meshes = self.config_instance.config["sdf_meshes"]

        # #为缺失的参数设置默认值
        # if "num_substeps" not in self.common:
        #     self.num_substeps = 1

    def get_common(self, key, default=None):
        if key in self.common:
            return self.common[key]
        else:
            logging.warning("Warning: key {} not found in common, return default value {}".format(key, default))
            return default
    
    def get_materials(self, key, default=None, id_=0,):
        if key in self.materials[id_]:
            return self.materials[id_][key]
        else:
            logging.warning("Warning: key {} not found in materials, return default value {}".format(key, default))
            return default
    
    def get_sdf_meshes(self, key, default=None, id_=0,):
        if not hasattr(self, "sdf_meshes"):
            logging.warning("Warning: sdf_meshes not found in config file, return None".format(None))
            return None
        if key in self.sdf_meshes[id_]:
            return self.sdf_meshes[id_][key]
        else:
            logging.warning("Warning: key {} not found in sdf_meshes, return default value {}".format(key, default))
            return default

meta = MetaData()
