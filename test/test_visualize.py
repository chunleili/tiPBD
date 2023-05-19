import sys, os
import taichi as ti

sys.path.append(os.getcwd())


def test_vis_sdf():
    from engine.visualize import vis_sdf
    from engine.sdf import SDF
    from engine.metadata import meta

    ti.init(arch=ti.cuda)
    meta.sdf_mesh_path = meta.get_common("sdf_mesh_path")
    sdf = SDF(meta.sdf_mesh_path, resolution=64)
    vis_sdf(sdf.val)


if __name__ == "__main__":
    test_vis_sdf()
