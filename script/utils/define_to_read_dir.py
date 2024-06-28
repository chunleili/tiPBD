import os, sys
import argparse

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
case_name = "soft3dBig"

parser = argparse.ArgumentParser()
parser.add_argument("-case_name", type=str, default=case_name)
args = parser.parse_args()
case_name = args.case_name

to_read_dir = prj_dir + f"result/{case_name}/A/"
print("to_read_dir", to_read_dir)