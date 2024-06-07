import pathlib
import shutil
import argparse

result_dir = pathlib.Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("-case_name", default= "latest")
parser.add_argument("-start_frame", default=0)
parser.add_argument("-end_frame", default=50)

args = parser.parse_args()
case_name = args.case_name
start_frame = args.start_frame
end_frame = args.end_frame

dir = result_dir / f"{case_name}" / "obj"

print(f"rename obj files (frame {start_frame} to {end_frame}) in {dir} from %04d.obj to %d.obj")

# change all %04d.obj to %d.obj
for i in range(start_frame, end_frame+1):
    src = dir / f"{i:04d}.obj"
    if not src.exists():
        continue
    dst = dir / f"{i}.obj"
    shutil.move(src, dst)
    print(f"rename {src.name} to {dst.name} done.")