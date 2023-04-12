def parse_cli_better_simple():
    import configargparse
    p = configargparse.ArgParser()
    p.add('-c', '--config_file', is_config_file=True, help='config file path', default='args.ini')
    p.add('--path', help='a file path') 
    p.add('--bool', help='a bool', action='store_true')

    options = p.parse_args()

    print(options)
    print("----------")
    print(p.format_help())
    print("----------")
    print(p.format_values())
    args = p.parse_args()
    pass

def parse_cli_better():
    import configargparse

    parser = configargparse.ArgumentParser(description='taichi PBD')
    parser.add_argument('-c', '--config_file', is_config_file=True, help='config file path', default='args.ini')
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
    args = parser.parse_args()
    
    import os
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    args.scene_file =  root_path + args.scene_file

    # print(args)
    print("----------")
    print(parser.format_help())
    print("----------")
    print(parser.format_values())

if __name__ == "__main__":
    parse_cli_better()