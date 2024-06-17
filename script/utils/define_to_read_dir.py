import os, sys
import argparse

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
case_name = "latest"
to_read_dir = prj_dir + f"result/{case_name}/A/"
print("to_read_dir", to_read_dir)