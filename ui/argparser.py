import argparse
from ui.config_builder import SimConfig
import os
from ui.filedialog import filedialog
from main import root_path
from engine.metadata import meta
def parse():
    parser = argparse.ArgumentParser(description='taichi PBD')

    parser.add_argument('--no-gui', action='store_true', default=False,
                        help='run without gui')
    args = parser.parse_args()
    no_gui = args.no_gui

    scene_path = filedialog()

    config = SimConfig(scene_file_path=scene_path)
    scene_name = scene_path.split("/")[-1].split(".")[0]
    return args, config, scene_name, no_gui