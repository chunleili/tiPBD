import subprocess, os
import scipy
import numpy as np

def test_export_mat():
    run_args = ["python", "engine/cloth/cloth3d.py","-N=64","-solver_type=AMG", "-end_frame=1", "-maxiter=2", "-export_matrix=1", "-export_matrix_dir=result/latest/A/", "-export_matrix_frame=1"]
    print("generating data...")
    ret = subprocess.call(run_args)
    assert ret == 0
    assert os.path.exists("result/latest/A/A_F1.npz")
    assert os.path.exists("result/latest/A/b_F1.npy")

    print("Loading A and b...")
    A = scipy.sparse.load_npz("A.npz")
    b = np.load("b.npy")
    print("A.shape", A.shape)
    print("b.shape", b.shape)
    assert A.shape[0] == b.shape[0]
    print("test_export_mat passed")


if __name__ == "__main__":
    test_export_mat()