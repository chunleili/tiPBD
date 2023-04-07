import json


class SimConfig:
    def __init__(self, scene_file_path) -> None:
        self.config = None
        with open(scene_file_path, "r") as f:
            self.config = json.load(f)
        print(json.dumps(self.config, indent=2))
    
    def get_common(self):
        if "common" in self.config:
            return self.config["common"]
        else:
            assert False, "No common in scene file"
    
    def get_materials(self):
        if "materials" in self.config:
            return self.config["materials"]
        else:
            assert False, "No materials in scene file"