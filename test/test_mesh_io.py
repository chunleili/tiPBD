import sys,os
import taichi as ti
import trimesh
sys.path.append(os.getcwd())
from engine.mesh_io import *
from engine.visualize import visualize

def test_points_from_volume():
    pts_np = points_from_volume()
    import taichi as ti
    ti.init()
    pts = ti.Vector.field(3, dtype=ti.f32, shape=pts_np.shape[0])
    pts.from_numpy(pts_np)
    visualize(pts)


def test_scale_to_unit_sphere():
    mesh = trimesh.load("data/model/bunny.obj")
    print("before scale")
    mesh = scale_to_unit_sphere(mesh)
    print("after scale")
    visualize(mesh.vertices, ti_init=True)

def test_scale_to_unit_cube():
    mesh = trimesh.load("data/model/bunny.obj")
    print("before scale")
    print(mesh.vertices.max(), mesh.vertices.min())
    mesh = scale_to_unit_cube(mesh)
    print("after scale")
    print(mesh.vertices.max(), mesh.vertices.min())
    visualize(mesh.vertices, ti_init=True)


def test_read_particles():
    pts = read_particles()
    visualize(pts, ti_init=True, show_widget=True)


if __name__=="__main__":
    # test_scale_to_unit_cube()
    # test_points_from_volume()
    test_read_particles()