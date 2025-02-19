#Convert the .node file so that it matches the format of painlessMG.
#The extra final column is added, which is all zeros.

import numpy as np
import sys,os
import tqdm

def convert(filename):
    """
    tiPBD to xian2019
    Args:
        filename: 网格文件名，不包含后缀名

    Returns:
        in-situ change the .node file
    """
    import numpy as np
    node_file_name = filename + ".node"

    with open(node_file_name, "r") as f:
        lines = f.readlines()
        NV = int(lines[0].split()[0])
        pos = np.zeros((NV, 3), dtype=np.float32)
        for i in range(NV):
            pos[i] = np.array(lines[i + 1].split()[1:], dtype=np.float32)

    with open(node_file_name, "w") as f:
        f.write(f"{pos.shape[0]} 3 0 1\n")
        for i in range(pos.shape[0]):
            f.write(f"{i} {pos[i, 0]} {pos[i, 1]} {pos[i, 2]} 0\n")
    print(f"Convert {node_file_name} done.")



def convert_back(filename):
    """
    xian2019 to tiPBD
    Args:
        filename: 网格文件名，不包含后缀名

    Returns:
        in-situ change the .node file
    """
    import numpy as np
    node_file_name = filename + ".node"

    with open(node_file_name, "r") as f:
        lines = f.readlines()
        NV = int(lines[0].split()[0])
        pos = np.zeros((NV, 3), dtype=np.float32)
        for i in range(NV):
            pos[i] = np.array(lines[i + 1].split()[1:4], dtype=np.float32)

    step_pbar = tqdm.tqdm(total=NV)
    with open(node_file_name, "w") as f:
        f.write(f"{pos.shape[0]} 3 0 0\n")
        for i in range(pos.shape[0]):
            step_pbar.update(1)
            f.write(f"{i} {pos[i, 0]} {pos[i, 1]} {pos[i, 2]}\n")
    print(f"Convert back {node_file_name} done.")


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python convert.py filename")
    #     sys.exit(1)
    # convert(sys.argv[1])
    print("Convert")
    convert_back("data/model/armadillo100K/Armadillo_100K.1")