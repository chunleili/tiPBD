import sys, os
import taichi as ti

sys.path.append(os.getcwd())
from engine.sdf import *
from engine.visualize import visualize, vis_sdf


def test_sdf_2d():
    sdf = SDF((5, 5))
    sdf.val.fill(1)
    print(sdf.val)
    print(sdf.grad)
    sdf.compute_gradient(1.0, 1.0)
    print(sdf)
    sdf.print_to_file()


def test_sdf_3d():
    sdf_3d = SDF((5, 5, 5))
    sdf_3d.val.fill(1)
    print(sdf_3d.val)
    print(sdf_3d.grad)
    sdf_3d.compute_gradient(1.0, 1.0, 1.0)
    print(sdf_3d)
    sdf_3d.print_to_file("result/sdf_3d")


def test_sdf_basic():
    # fill with 1
    sdf = SDF(None, 5, dim=2)
    sdf.val.fill(1)
    sdf.compute_gradient(1.0, 1.0)
    print(sdf)

    # fill with 1, 3d
    sdf_3d = SDF(None, 5, dim=3)
    sdf_3d.val.fill(1)
    sdf_3d.compute_gradient(1.0, 1.0, 1.0)
    print(sdf_3d)

    # random fill, 3d
    import numpy as np

    sdf_3d = SDF(None, 5, dim=3)
    sdf_3d.val.from_numpy(np.random.rand(5, 5, 5))
    sdf_3d.compute_gradient()
    print(sdf_3d)


def test_gen_sdf_points():
    val, grad = gen_sdf_points()
    visualize(val)


def test_gen_sdf_voxels():
    vox, grad = gen_sdf_voxels("data/model/chair.obj", 64, True)
    vis_sdf(vox)


def test_SDF():
    sdf = SDF("data/model/chair.obj", 64, 3)
    vis_sdf(sdf.val)


if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    # test_sdf_basic()
    # test_gen_sdf_voxels()
    test_SDF()
    # test_sdf_2d()
    # test_sdf_3d()
