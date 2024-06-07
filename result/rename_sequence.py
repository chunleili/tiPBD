import pathlib
import shutil


result_dir = pathlib.Path(__file__).resolve().parent

dir = result_dir / "scale64-10086" / "obj"

print(f"rename all obj files in {dir} from %04d.obj to %d.obj")

start_frame = 0
end_frame = 50

# change all %04d.obj to %d.obj
for i in range(start_frame, end_frame+1):
    src = dir / f"{i:04d}.obj"
    if not src.exists():
        continue
    dst = dir / f"{i}.obj"
    shutil.move(src, dst)
    print(f"rename {src.name} to {dst.name} done.")