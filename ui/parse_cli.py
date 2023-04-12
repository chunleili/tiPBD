def parse_cli():
    '''
    Read command line arguments
    '''
    import argparse
    parser = argparse.ArgumentParser(description='taichi PBD')
    parser.add_argument('--scene_file', type=str, default="",
                        help='manually specify scene file, if not specified, use gui to select')
    parser.add_argument('--no-gui', action='store_true', default=False,
                        help='no gui mode')
    parser.add_argument("--arch", type=str, default="cuda",
                        help="backend(arch) of taichi)")
    parser.add_argument("--kernel_profiler", action='store_true', default=False,
                        help="enable kernel profiler")
    parser.add_argument("--use_dearpygui", action='store_true', default=False,
                        help="use dearpygui as gui")
    parser.add_argument("--debug", action='store_true', default=False,
                    help="debug mode")
    parser.add_argument("--device_memory_GB", type=float, default=0.5,
                    help="device memory in GB")
    args = parser.parse_args()

    import os
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # args.scene_file =  root_path+"/data/scene/bunny_fluid.json"
    # args.device_memory_GB = 3.0

    import taichi as ti
    if args.arch == "cuda":
        args.arch = ti.cuda
    elif args.arch == "gpu":
        args.arch = ti.gpu
    elif args.arch == "cpu":
        args.arch = ti.cpu
    else:
        args.arch = None

    return args