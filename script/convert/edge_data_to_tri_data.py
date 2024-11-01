import numpy as np
import json
import tqdm
import sys,os

sys.path.append(os.getcwd())
from engine.mesh_io import edge_data_to_tri_data

def edge_data_to_tri_data_batch(dir, frames):
    # E:\Dev\tiPBD\result\latest\mesh\0005_strain.txt
    # read e2t,  tri
    e2t = json.load(open(dir+"/e2t.json"))

    # change key from string to int
    e2t = {int(k): v for k, v in e2t.items()}

    tri = np.loadtxt(dir+"/tri.txt", dtype=np.int32)
    # read edge data
    import tqdm
    step_pbar = tqdm.tqdm(total=len(frames))
    for f in frames:
        edge_data = np.loadtxt(dir+f"{f:04d}_strain.txt",skiprows=1)
        tri_data = edge_data_to_tri_data(e2t, edge_data, tri)
        np.savetxt(dir+f"{f:04d}_strain_tri.txt", tri_data)
        step_pbar.update(1)
    step_pbar.close()
    print("Output to ", dir)
    print("done")
        

if __name__ == "__main__":
    edge_data_to_tri_data_batch("E:/Dev/tiPBD/result/latest/mesh/", range(1, 10))