import pathlib
import shutil,os
import argparse

result_dir = pathlib.Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("-case_name", default= "latest")
parser.add_argument("-start_frame",  type=int, default=1)
parser.add_argument("-end_frame", type=int, default=600)
parser.add_argument("-inc",  type=int, default=3)

args = parser.parse_args()
case_name = args.case_name
start_frame = args.start_frame
end_frame = args.end_frame
inc = args.inc

dir = result_dir 

print(f"rename png files (frame {start_frame} to {end_frame}) in {dir} from %04d.png to %d.png")

shutil.copytree(dir, str(dir)+"_bak", dirs_exist_ok=True)

# change all %04d.png to %d.png
for i in range(start_frame, end_frame+1, inc):
    src = dir / f"{i:04d}.png"
    if not src.exists():
        continue
    dst = dir / f"{i:04d}.png"
    shutil.move(src, dst)
    print(f"rename {src.name} to {dst.name} done.")