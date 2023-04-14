import json


class SimConfig:
    def __init__(self, scene_file_path) -> None:
        self.config = None
        with open(scene_file_path, "r") as f:
            self.config = json.load(f)
        print(json.dumps(self.config, indent=2))
    
    # def get_common(self, key, default=None):
    #     if key in self.config['common']:
    #         return self.config['common'][key]
    #     else:
    #         return default
        
    # def get_materials(self, key, default=None):
    #     if key in self.config["materials"]:
    #         return self.config["materials"][key]
    #     else:
    #         return default