import re
from pathlib import Path

def create_another_outdir(out_dir):
    path = Path(out_dir)
    if path.exists():
        # 使用正则表达式匹配文件夹名称中的数字后缀
        base_name = path.name
        match = re.search(r'_(\d+)$', base_name)
        if match:
            base_name = base_name[:match.start()]
            i = int(match.group(1)) + 1
        else:
            base_name = base_name
            i = 1

        while True:
            new_name = f"{base_name}_{i}"
            path = path.parent / new_name
            if not path.exists():
                break
            i += 1

    path.mkdir(parents=True, exist_ok=True)
    out_dir = str(path)
    print(f"\ncreate another outdir: {out_dir}\n")
    return out_dir

def main():
    out_dir = "result/testDir"
    out_dir = create_another_outdir(out_dir)

if __name__ == "__main__":
    main()
