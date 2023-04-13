import taichi as ti

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
        
        self.frame = 0
        self.step = 0

        self.args = parse_cli()
        print("args:", self.args)

        if self.args.scene_file is None:
            self.scene_path = filedialog()
        else:
            self.scene_path = self.args.scene_file
        self.scene_name = self.scene_path.split("/")[-1].split(".")[0]
        self.config = SimConfig(scene_file_path=self.scene_path)
        self.common = self.config.get_common()
        self.materials = self.config.get_materials()

meta = MetaData()
