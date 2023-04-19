def parse_cli():
    import configargparse
    import taichi as ti
    import os

    parser = configargparse.ArgumentParser(description='taichi PBD')
    parser.add_argument('--scene_file', type=str, default="",
                        help='manually specify scene file, if not specified, use gui to select')
    parser.add_argument('--no-gui', action='store_true', default=False,
                        help='no gui mode')
    parser.add_argument("--arch", type=str, default="cuda",
                        help="backend(arch) of taichi)")
    parser.add_argument("--debug", action='store_true', default=False,
                    help="debug mode")
    parser.add_argument("--device_memory_fraction", type=float, default=0.5,
                    help="device memory fraction")
    parser.add_argument("--kernel_profiler", action='store_true', default=False,
                        help="enable kernel profiler")
    parser.add_argument("--use_dearpygui", action='store_true', default=False,
                        help="use dearpygui as gui")
    parser.add_argument("--use_webui", action='store_true', default=False,
                        help="use webui")
    parser.add_argument("--use_solver_main", action='store_true', default=True,
                        help="use provided solver_main")
    parser.add_argument('-c', '--config_file', is_config_file=True,
                         help='config file path', default='args.ini')
    args = parser.parse_args()
    
    if args.scene_file != "":
        root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        args.scene_file =  root_path + args.scene_file

    # print("----------")
    # print(parser.format_help())
    # print("----------")
    print(parser.format_values())

    if args.arch == "cuda":
        args.arch = ti.cuda
    elif args.arch == "cpu":
        args.arch = ti.cpu
    elif args.arch == "gpu":
        args.arch = ti.gpu
    else:
        args.arch = None

    # 把init_args打包， ti.init(**args.init_args)
    args.init_args = {"arch": args.arch, "device_memory_fraction": args.device_memory_fraction, "kernel_profiler": args.kernel_profiler, "debug": args.debug}

    # print(args)
    return args



# def parse_cli(): # old version, use built-in argparse
#     '''
#     Read command line arguments
#     '''
#     import argparse
#     import taichi as ti
#     import os
#     parser = argparse.ArgumentParser(description='taichi PBD')
#     parser.add_argument('--scene_file', type=str, default="",
#                         help='manually specify scene file, if not specified, use gui to select')
#     parser.add_argument('--no-gui', action='store_true', default=False,
#                         help='no gui mode')
#     parser.add_argument("--arch", type=str, default="cuda",
#                         help="backend(arch) of taichi)")
#     parser.add_argument("--debug", action='store_true', default=False,
#                     help="debug mode")
#     parser.add_argument("--device_memory_fraction", type=float, default=0.5,
#                     help="device memory fraction")
#     parser.add_argument("--kernel_profiler", action='store_true', default=False,
#                         help="enable kernel profiler")
#     parser.add_argument("--use_dearpygui", action='store_true', default=False,
#                         help="use dearpygui as gui")
#     args = parser.parse_args()

#     # root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
#     # args.scene_file =  root_path+"/data/scene/bunny_fluid.json"

#     if args.arch == "cuda":
#         args.arch = ti.cuda
#     elif args.arch == "cpu":
#         args.arch = ti.cpu
#     else:
#         args.arch = None

#     # 把init_args打包， ti.init(**args.init_args)
#     args.init_args = {"arch": args.arch, "device_memory_fraction": args.device_memory_fraction, "kernel_profiler": args.kernel_profiler, "debug": args.debug}
#     return args