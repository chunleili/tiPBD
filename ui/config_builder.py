import json


class SimConfig:
    def __init__(self, scene_file_path) -> None:
        self.config = None
        with open(scene_file_path, "r") as f:
            self.config = json.load(f)
        print(self.config)
    
    def get_common(self):
        if "common" in self.config:
            return self.config["common"]
        else:
            assert False, "No common in scene file"
    
    def get_solids(self):
        if "solids" in self.config:
            return self.config["solids"]
        else:
            assert False, "No solids in scene file"
