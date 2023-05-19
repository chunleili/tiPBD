import taichi as ti
import os
import numpy as np
import logging


@ti.data_oriented
class SDF:
    """
    Signed Distance Field (SDF) class.
    """

    def __init__(self, mesh_path=None, resolution=64, dim=3, use_cache=True):
        """
        生成SDF体素场。其中有两个taichi field: val和grad，分别表示SDF体素场的值和梯度。

        Args:
            mesh_path (str): 网格文件路径，如果为None则需要手动填充SDF体素场。
            resolution (int): SDF体素场分辨率。默认为64。
            dim (int): SDF体素场维度。默认为3。
            use_cache (bool): 是否使用缓存。默认为True。
        """
        self.resolution = resolution
        self.dim = dim
        if dim == 2:
            self.shape = (resolution, resolution)
        elif dim == 3:
            self.shape = (resolution, resolution, resolution)
        else:
            raise Exception("SDF only supports 2D/3D for now")
        print("SDF shape = ", self.shape)
        print("SDF initilizing...")

        from engine.metadata import meta

        meta.use_sparse = meta.get_sdf_meshes("use_sparse", False)
        meta.narrow_band = meta.get_sdf_meshes("narrow_band", 0)

        if meta.use_sparse:
            print("Using sparse SDF...")
            self.val = ti.field(dtype=ti.f32)
            self.grad = ti.Vector.field(self.dim, dtype=ti.f32)
            if dim == 2:
                self.snode = ti.root.bitmasked(ti.ij, self.shape)
            elif dim == 3:
                self.snode = ti.root.bitmasked(ti.ijk, self.shape)
            self.snode.place(self.val)
            self.snode.place(self.grad)
        else:
            self.val = ti.field(dtype=ti.f32, shape=self.shape)
            self.grad = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.shape)

        if mesh_path is not None:
            print(f"sdf mesh path: {mesh_path}")
            # 检查是否存在cache
            val_cache_path = mesh_path + "_sdf_cache_val.npy"
            grad_cache_path = mesh_path + "_sdf_cache_grad.npy"
            if not os.path.exists(val_cache_path) or not os.path.exists(grad_cache_path) or not use_cache:
                print("No sdf cache found. Generating sdf cache...")
                val_np, grad_np = gen_sdf_voxels(mesh_path, resolution, True)
                np.save(val_cache_path, val_np)
                np.save(grad_cache_path, grad_np)
            else:
                print("Found sdf cache. Loading sdf cache...")
                val_np = np.load(val_cache_path)
                grad_np = np.load(grad_cache_path)

        self.val.from_numpy(val_np)
        self.grad.from_numpy(grad_np)

        if meta.narrow_band > 0 and meta.use_sparse:
            print("Using narrow band...(Only when use_sparse is True)")
            make_narrow_band(self, meta.narrow_band, resolution)

        print("SDF init done.")


@ti.kernel
def make_narrow_band(sdf: ti.template(), narrow_band: int, resolution: ti.i32):
    dx = 1.0 / sdf.resolution
    for I in ti.grouped(sdf.val):
        if sdf.val[I] > narrow_band * dx:
            ti.deactivate(sdf.snode, I)
        elif sdf.val[I] < -narrow_band * dx:
            ti.deactivate(sdf.snode, I)


def gen_sdf_voxels(mesh_path, resolution=64, return_gradients=False):
    """
    从表面网格生成体素(靠近网格表面处的)SDF场。借助mesh_to_sdf库和trimesh。注意导入的模型会自动缩放到[-1,1]的立方体内。

    Args:
        mesh_path (str, optional): 网格文件路径。 Defaults to 'data/model/chair.obj'.
        resolution (int, optional): 体素分辨率。 Defaults to 64.
    """
    import trimesh

    mesh = trimesh.load(mesh_path)
    vox = mesh_to_voxels(mesh, voxel_resolution=resolution, return_gradients=return_gradients)
    return vox


def mesh_to_voxels(
    mesh,
    voxel_resolution=64,
    surface_point_method="scan",
    sign_method="normal",
    scan_count=100,
    scan_resolution=400,
    sample_point_count=10000000,
    normal_sample_count=11,
    pad=False,
    check_result=False,
    return_gradients=False,
):
    from mesh_to_sdf import get_surface_point_cloud
    from engine.mesh_io import scale_to_unit_cube
    from engine.metadata import meta

    mesh = scale_to_unit_cube(mesh)
    s = meta.get_sdf_meshes("scale")
    t = meta.get_sdf_meshes("translation")
    if s is not None:
        mesh.apply_scale(s)
    if t is not None:
        mesh.apply_translation(t)

    surface_point_cloud = get_surface_point_cloud(
        mesh, surface_point_method, 3**0.5, scan_count, scan_resolution, sample_point_count, sign_method == "normal"
    )

    return surface_point_cloud.get_voxels(
        voxel_resolution, sign_method == "depth", normal_sample_count, pad, check_result, return_gradients
    )


@ti.func
def collision_response(pos: ti.template(), sdf):
    sdf_epsilon = 1e-4
    grid_idx = ti.Vector([pos.x * sdf.resolution, pos.y * sdf.resolution, pos.z * sdf.resolution], ti.i32)
    normal = sdf.grad[grid_idx]
    sdf_val = sdf.val[grid_idx]
    assert normal.norm() == 1.0
    if sdf_val < sdf_epsilon:
        pos -= sdf_val * normal
        # if vel.dot(normal) < 0:
        #     normal_component = normal.dot(vel)
        #     vel -=  normal * min(normal_component, 0)
