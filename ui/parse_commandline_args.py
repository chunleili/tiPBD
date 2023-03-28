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
    args = parser.parse_args()

    import taichi as ti
    if args.arch == "cuda":
        args.arch = ti.cuda
    elif args.arch == "x64":
        args.arch = ti.x64
    else:
        args.arch = None

    return args