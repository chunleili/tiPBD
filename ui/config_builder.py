import json


class SimConfig:
    def __init__(self, scene_file_path) -> None:
        self.config = None
        with open(scene_file_path, "r") as f:
            self.config = json.load(f)
        print(self.config)
    
    def get_cfg(self, name, enforce_exist=False):
        if enforce_exist:
            assert name in self.config["common"]
        if name not in self.config["common"]:
            if enforce_exist:
                assert name in self.config["common"]
            else:
                return None
        return self.config["common"][name]
    
    def get_solids(self):
        if "solids" in self.config:
            return self.config["solids"]
        else:
            assert False, "No solids in scene file"
