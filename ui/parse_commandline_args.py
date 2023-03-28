import argparse
'''
Read command line arguments
'''
def parse_commandline_args():
    parser = argparse.ArgumentParser(description='taichi PBD')
    parser.add_argument('--use_scene_file', action='store_true', default=True,
                        help='use scene file to read parameters')
    args = parser.parse_args()
    return args