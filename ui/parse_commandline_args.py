import argparse
'''
Read command line arguments
'''
def parse_commandline_args():
    parser = argparse.ArgumentParser(description='taichi PBD')

    parser.add_argument('--use_scene_file', action='store_true', default=True,
                        help='use scene file to read parameters')
    parser.add_argument('--no-gui', action='store_true', default=False,
                        help='no gui mode')
    parser.add_argument("--arch", type=str, default="cuda",
                        help="backend(arch) of taichi)")
    parser.add_argument("--kernel_profiler", action='store_true', default=False,
                        help="enable kernel profiler")
    parser.add_argument("--use_dearpygui", action='store_true', default=False,
                        help="use dearpygui as gui")
    args = parser.parse_args()

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