import pathlib
import shutil,os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-dir", default= "")
parser.add_argument("-start",  type=int, default=1)
parser.add_argument("-end", type=int, default=600)
parser.add_argument("-inc",  type=int, default=3)

args = parser.parse_args()

if args.dir == "":
    dir = pathlib.Path(__file__).resolve().parent
else:
    dir = pathlib.Path(args.dir)

start_frame = args.start
end_frame = args.end
inc = args.inc

print(f"rename png files (frame {start_frame} to {end_frame}) in {dir} from %04d.png to %d.png")

print("backup start...")
shutil.copytree(dir, str(dir)+"_bak", dirs_exist_ok=True)
print("backup done.")

# change all %04d.png to %d.png
for i in range(start_frame, end_frame+1, inc):
    src = dir / f"{i:04d}.png"
    print(f"src: {src}")
    if not src.exists():
        continue
    dst = dir / f"{(i+2)//3:04d}.png"
    shutil.move(src, dst)
    print(f"rename {src.name} to {dst.name} done.")