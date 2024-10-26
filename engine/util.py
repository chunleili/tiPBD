import taichi as ti
import numpy as np    
import logging
import json
import os


def filedialog():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.filename = filedialog.askopenfilename(initialdir="data/scene", title="Select a File")
    filename = root.filename
    root.destroy()  # close the window
    print("Open scene file: ", filename)
    return filename



def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner


class SimConfig:
    def __init__(self, scene_file_path) -> None:
        self.config = None
        with open(scene_file_path, "r") as f:
            self.config = json.load(f)
        print(json.dumps(self.config, indent=2))

    # def get_common(self, key, default=None):
    #     if key in self.config['common']:
    #         return self.config['common'][key]
    #     else:
    #         return default

    # def get_materials(self, key, default=None):
    #     if key in self.config["materials"]:
    #         return self.config["materials"][key]
    #     else:
    #         return default



############################################
# parse cli

def parse_cli(): # old version, use built-in argparse
    '''
    Read command line arguments
    '''
    import argparse
    import taichi as ti
    import os
    parser = argparse.ArgumentParser(description='taichi PBD')
    parser.add_argument('--scene_file', type=str, default="",
                        help='manually specify scene file, if not specified, use gui to select')
    parser.add_argument('--no_gui', action='store_true', default=False,
                        help='no gui mode')
    parser.add_argument("--arch", type=str, default="cuda",
                        help="backend(arch) of taichi)")
    parser.add_argument("--debug", action='store_true', default=False,
                    help="debug mode")
    parser.add_argument("--device_memory_fraction", type=float, default=0.5,
                    help="device memory fraction")
    parser.add_argument("--kernel_profiler", action='store_true', default=False,
                        help="enable kernel profiler")
    parser.add_argument("--use_dearpygui", action='store_true', default=False,
                        help="use dearpygui as gui")
    args = parser.parse_args()

    # root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # args.scene_file =  root_path+"/data/scene/bunny_fluid.json"

    if args.arch == "cuda":
        args.arch = ti.cuda
    elif args.arch == "cpu":
        args.arch = ti.cpu
    else:
        args.arch = None

    # 把init_args打包， ti.init(**args.init_args)
    args.init_args = {"arch": args.arch, "device_memory_fraction": args.device_memory_fraction, "kernel_profiler": args.kernel_profiler, "debug": args.debug}
    return args



@singleton
@ti.data_oriented
class MetaData:
    def __init__(self):
        import os
        # from engine.util import parse_cli

        self.root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.result_path = os.path.join(self.root_path, "result")
        print("root_path:", self.root_path)

        self.args = parse_cli()
        # print("args:", self.args)

        self.common = {}
        self.materials = [{}]

        # if self.args.no_json == True:
        #     return

        if self.args.scene_file == "":

            self.scene_path = filedialog()
        else:
            self.scene_path = self.args.scene_file
        self.scene_name = self.scene_path.split("/")[-1].split(".")[0]


        self.config_instance = SimConfig(scene_file_path=self.scene_path)
        self.common = self.config_instance.config["common"]
        self.materials = self.config_instance.config["materials"]
        if "sdf_meshes" in self.config_instance.config:
            self.sdf_meshes = self.config_instance.config["sdf_meshes"]

        # #为缺失的参数设置默认值
        # if "num_substeps" not in self.common:
        #     self.num_substeps = 1

    def get_common(self, key, default=None, no_warning=False):
        if key in self.common:
            return self.common[key]
        else:
            if not no_warning:
                logging.warning("key {} not found in common, use default value {}".format(key, default))
            return default

    def get_materials(self, key, default=None, id_=0, no_warning=False):
        if key in self.materials[id_]:
            return self.materials[id_][key]
        else:
            if not no_warning:
                logging.warning("key {} not found in materials, use default value {}".format(key, default))
            return default

    def get_sdf_meshes(self, key, default=None, id_=0, no_warning=False):
        if not hasattr(self, "sdf_meshes"):
            if not no_warning:
                logging.warning("sdf_meshes not found in config file, return None".format(None))
            return None
        if key in self.sdf_meshes[id_]:
            return self.sdf_meshes[id_][key]
        else:
            if not no_warning:
                logging.warning("key {} not found in sdf_meshes, use default value {}".format(key, default))
            return default


# meta = MetaData()


@ti.kernel
def init_tet_indices(mesh: ti.template(), indices: ti.template()):
    for c in mesh.cells:
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id


def field_from_numpy(x_np):
    import numpy as np
    import taichi as ti

    ti.init()
    x = ti.Vector.field(3, dtype=ti.f32, shape=x_np.shape[0])
    x.from_numpy(x_np)
    return x


def np_to_ti(input, dim=1):
    import numpy as np

    if isinstance(input, np.ndarray):
        if dim == 1:
            out = ti.field(dtype=ti.f32, shape=input.shape)
            out.from_numpy(input)
        else:
            out = ti.Vector.field(dim, dtype=ti.f32, shape=input.shape)
            out.from_numpy(input)
    else:
        out = input
    return out


@ti.kernel
def random_fill_vec(x: ti.template(), dim: ti.template()):
    shape = x.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for d in ti.static(range(dim)):
                    x[i, j, k][d] = ti.random()


@ti.kernel
def random_fill_scalar(x: ti.template()):
    shape = x.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                x[i, j, k] = ti.random()


def random_fill(x, dim):
    if dim > 1:
        random_fill_vec(x, dim)
    else:
        random_fill_scalar(x)


def print_to_file(val, filename="field", dim=3):
    import numpy as np

    val = val.to_numpy()
    if dim == 1:
        np.savetxt("result/" + filename + ".csv", val, fmt="%.2e")
    else:
        np.savetxt("result/" + filename + ".csv", val.reshape(-1, dim), fmt="%.2e")


@ti.func
def pos_to_grid_idx(x, y, z, dx, dy, dz):
    return ti.Vector([x / dx, y / dy, z / dz], ti.i32)





############################################
# gradient

vec2 = ti.types.vector(2, dtype=ti.f32)
vec3 = ti.types.vector(3, dtype=ti.f32)

@ti.func
def grad_at_xy(val, x, y, dx, dy) -> ti.math.vec2:
    """
    Compute the gradient of a 2d scalar field at a given position(x,y).

    Args:
        val (ti.template()): 2D scalar field
        x (ti.f32): x position
        y (ti.f32): y position
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction

    Returns:
        vec2: gradient at (x,y)

    """
    i = int(x)
    j = int(y)
    u = x - i
    v = y - j

    grad00 = grad_at_ij(val, i, j, dx, dy)
    grad01 = grad_at_ij(val, i, j + 1, dx, dy)
    grad10 = grad_at_ij(val, i + 1, j, dx, dy)
    grad11 = grad_at_ij(val, i + 1, j + 1, dx, dy)
    res = (1 - u) * (1 - v) * grad00 + u * (1 - v) * grad10 + (1 - u) * v * grad01 + u * v * grad11
    return res


@ti.func
def grad_at_xyz(val, x, y, z, dx, dy, dz) -> ti.math.vec3:
    """
    Compute the gradient of a 3d scalar field at a given position(x,y,z).

    Args:
        val (ti.template()): 3D scalar field
        x (ti.f32): x position
        y (ti.f32): y position
        z (ti.f32): z position
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        dz (ti.f32): grid spacing in z direction

    Returns:
        vec3: gradient at (x,y,z)
    """
    i = int(x)
    j = int(y)
    k = int(z)
    u = x - i
    v = y - j
    w = z - k

    grad000 = grad_at_ijk(val, i, j, k, dx, dy, dz)
    grad001 = grad_at_ijk(val, i, j, k + 1, dx, dy, dz)
    grad010 = grad_at_ijk(val, i, j + 1, k, dx, dy, dz)
    grad011 = grad_at_ijk(val, i, j + 1, k + 1, dx, dy, dz)
    grad100 = grad_at_ijk(val, i + 1, j, k, dx, dy, dz)
    grad101 = grad_at_ijk(val, i + 1, j, k + 1, dx, dy, dz)
    grad110 = grad_at_ijk(val, i + 1, j + 1, k, dx, dy, dz)
    grad111 = grad_at_ijk(val, i + 1, j + 1, k + 1, dx, dy, dz)
    res = (
        (1 - u) * (1 - v) * (1 - w) * grad000
        + u * (1 - v) * (1 - w) * grad100
        + (1 - u) * v * (1 - w) * grad010
        + u * v * (1 - w) * grad110
        + (1 - u) * (1 - v) * w * grad001
        + u * (1 - v) * w * grad101
        + (1 - u) * v * w * grad011
        + u * v * w * grad111
    )
    return res


def bilinear_weight(x, y):
    """
    Bilinear sample weights of a 2D scalar field at a given position.
    """
    i = int(x)
    j = int(y)
    u = x - i
    v = y - j
    return (1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v


def trilinear_weight(x, y, z):
    """
    Trilinear sample weights of a 3D scalar field at a given position.
    """
    i = int(x)
    j = int(y)
    k = int(z)
    u = x - i
    v = y - j
    w = z - k
    return (
        (1 - u) * (1 - v) * (1 - w),
        u * (1 - v) * (1 - w),
        (1 - u) * v * (1 - w),
        u * v * (1 - w),
        (1 - u) * (1 - v) * w,
        u * (1 - v) * w,
        (1 - u) * v * w,
        u * v * w,
    )


def compute_all_gradient(val, dx, dy, dz=None):
    """
    Compute all the gradient of the SDF field.

    this will automatically determine the dimension of the field and call the corresponding function.
    """
    shape = val.shape
    dim = len(shape)
    if dim == 2:
        res = compute_all_gradient_2d(val, dx, dy)
    elif dim == 3:
        res = compute_all_gradient_3d(val, dx, dy, dz)
    else:
        raise ValueError(f"Only support 2D and 3D, but got {dim}D")
    return res


@ti.kernel
def compute_all_gradient_2d(val: ti.template(), dx: ti.f32, dy: ti.f32, grad: ti.template()):
    """
    Using central difference to compute all gradients of a 2D scalar field

    Args:
        val (ti.template()): 2D scalar field
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        grad (ti.template()): 2D vector field (result field)
    """
    shape = val.shape
    dx_inv, dy_inv = 1.0 / dx, 1.0 / dy
    for i, j in val:
        if 0 < i < shape[0] - 1 and 0 < j < shape[1] - 1:
            grad[i, j] = ti.Vector(
                [(val[i + 1, j] - val[i - 1, j]) * dx_inv * 0.5, (val[i, j + 1] - val[i, j - 1]) * dy_inv * 0.5]
            )


@ti.kernel
def compute_all_gradient_3d(val: ti.template(), dx: ti.f32, dy: ti.f32, dz: ti.f32, grad: ti.template()):
    """
    Using central difference to compute all gradients of a 3D scalar field

    Args:
        val (ti.template()): 3D scalar field
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        dz (ti.f32): grid spacing in z direction
        grad (ti.template()): 3D vector field (result field)
    """
    shape = val.shape
    dx_inv, dy_inv, dz_inv = 1.0 / dx, 1.0 / dy, 1.0 / dz
    for i, j, k in val:
        if 0 < i < shape[0] - 1 and 0 < j < shape[1] - 1 and 0 < k < shape[2] - 1:
            grad[i, j, k] = ti.Vector(
                [
                    (val[i + 1, j, k] - val[i - 1, j, k]) * dx_inv * 0.5,
                    (val[i, j + 1, k] - val[i, j - 1, k]) * dy_inv * 0.5,
                    (val[i, j, k + 1] - val[i, j, k - 1]) * dz_inv * 0.5,
                ]
            )


@ti.func
def grad_at_ij(val: ti.template(), dx: ti.f32, dy: ti.f32, i: ti.i32, j: ti.i32) -> vec2:
    """
    Using central difference to compute the gradient of a 2D scalar field at a given point

    Args:
        val (ti.template()): 2D scalar field
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        i (ti.i32): x index of the point
        j (ti.i32): y index of the point

    Returns:
        vec2: gradient at the given point
    """
    shape = val.shape
    res = ti.Vector([0.0, 0.0])
    if i == 0:
        res[0] = (val[1, j] - val[0, j]) / dx
    elif i == shape[0] - 1:
        res[0] = (val[shape[0] - 1, j] - val[shape[0] - 2, j]) / dx
    else:
        res[0] = (val[i + 1, j] - val[i - 1, j]) / dx * 0.5
    if j == 0:
        res[1] = (val[i, 1] - val[i, 0]) / dy
    elif j == shape[1] - 1:
        res[1] = (val[i, shape[1] - 1] - val[i, shape[1] - 2]) / dy
    else:
        res[1] = (val[i, j + 1] - val[i, j - 1]) / dy * 0.5
    return res


@ti.func
def grad_at_ijk(val: ti.template(), dx: ti.f32, dy: ti.f32, dz: ti.f32, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:
    """
    Using central difference to compute the gradient of a 3D scalar field at a given point

    Args:
        val (ti.template()): 3D scalar field
        dx (ti.f32): grid spacing in x direction
        dy (ti.f32): grid spacing in y direction
        dz (ti.f32): grid spacing in z direction
        i (ti.i32): x index of the point
        j (ti.i32): y index of the point
        k (ti.i32): z index of the point

    Returns:
        vec3: gradient at the given point
    """
    shape = val.shape
    res = ti.Vector([0.0, 0.0, 0.0])
    if i == 0:
        res[0] = (val[1, j, k] - val[0, j, k]) / dx
    elif i == shape[0] - 1:
        res[0] = (val[shape[0] - 1, j, k] - val[shape[0] - 2, j, k]) / dx
    else:
        res[0] = (val[i + 1, j, k] - val[i - 1, j, k]) / dx * 0.5
    if j == 0:
        res[1] = (val[i, 1, k] - val[i, 0, k]) / dy
    elif j == shape[1] - 1:
        res[1] = (val[i, shape[1] - 1, k] - val[i, shape[1] - 2, k]) / dy
    else:
        res[1] = (val[i, j + 1, k] - val[i, j - 1, k]) / dy * 0.5
    if k == 0:
        res[2] = (val[i, j, 1] - val[i, j, 0]) / dz
    elif k == shape[2] - 1:
        res[2] = (val[i, j, shape[2] - 1] - val[i, j, shape[2] - 2]) / dz
    else:
        res[2] = (val[i, j, k + 1] - val[i, j, k - 1]) / dz * 0.5


@ti.func
def bilinear_interpolate(val: ti.template(), x: ti.f32, y: ti.f32) -> ti.f32:
    """
    Bilinear interpolation of a 2D scalar field

    Args:
        val (ti.template()): 2D scalar field
        x (ti.f32): x coordinate of the point
        y (ti.f32): y coordinate of the point

    Returns:
        ti.f32: interpolated value
    """
    shape = val.shape
    i = int(x)
    j = int(y)
    if i < 0 or i >= shape[0] - 1 or j < 0 or j >= shape[1] - 1:
        return 0.0
    s = x - i
    t = y - j
    return (
        (1 - s) * (1 - t) * val[i, j]
        + s * (1 - t) * val[i + 1, j]
        + (1 - s) * t * val[i, j + 1]
        + s * t * val[i + 1, j + 1]
    )


@ti.func
def trilinear_interpolate(val: ti.template(), x: ti.f32, y: ti.f32, z: ti.f32) -> ti.f32:
    """
    Trilinear interpolation of a 3D scalar field

    Args:
        val (ti.template()): 3D scalar field
        x (ti.f32): x coordinate of the point
        y (ti.f32): y coordinate of the point
        z (ti.f32): z coordinate of the point

    Returns:
        ti.f32: interpolated value
    """
    shape = val.shape
    i = int(x)
    j = int(y)
    k = int(z)
    if i < 0 or i >= shape[0] - 1 or j < 0 or j >= shape[1] - 1 or k < 0 or k >= shape[2] - 1:
        return 0.0
    s = x - i
    t = y - j
    u = z - k
    return (
        (1 - s) * (1 - t) * (1 - u) * val[i, j, k]
        + s * (1 - t) * (1 - u) * val[i + 1, j, k]
        + (1 - s) * t * (1 - u) * val[i, j + 1, k]
        + s * t * (1 - u) * val[i + 1, j + 1, k]
        + (1 - s) * (1 - t) * u * val[i, j, k + 1]
        + s * (1 - t) * u * val[i + 1, j, k + 1]
        + (1 - s) * t * u * val[i, j + 1, k + 1]
        + s * t * u * val[i + 1, j + 1, k + 1]
    )





############################################
# sdf

@ti.data_oriented
class SDF:
    """
    Signed Distance Field (SDF) class.
    """

    def __init__(self, meta, mesh_path=None, resolution=64, dim=3, use_cache=True):
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
    meta,
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



############################################
# p2g
   
@ti.kernel
def p2g_2d(x: ti.template(), dx: ti.f32, grid_m: ti.template()):
    """
    将周围粒子的质量scatter到2D网格上。实际上，只要替换grid_m, 可以scatter任何标量场。

    Args:
        x (ti.template()): 粒子位置
        dx (ti.f32): 网格间距
        grid_m (ti.template()): 网格质量(输出)
    """
    inv_dx = 1.0 / dx
    p_mass = 1.0
    for p in x:
        base = ti.cast(ti.floor(x[p] * inv_dx - 0.5), ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                I = ti.Vector([i, j])
                weight = w[i].x * w[j].y
                if in_bound(base, I, grid_m.shape):
                    grid_m[base + I] += weight * p_mass


@ti.kernel
def p2g_3d(x: ti.template(), dx: ti.f32, grid_m: ti.template()):
    """
    将周围粒子的质量scatter到3D网格上。实际上，只要替换grid_m, 可以scatter任何标量场。

    Args:
        x (ti.template()): 粒子位置
        dx (ti.f32): 网格间距
        grid_m (ti.template()): 网格质量(输出)
    """
    inv_dx = 1.0 / dx
    p_mass = 1.0
    shape = grid_m.shape
    for p in x:
        base = ti.cast(ti.floor(x[p] * inv_dx - 0.5), ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    I = ti.Vector([i, j, k])
                    weight = w[i].x * w[j].y * w[k].z
                    if in_bound(base, I, shape):
                        grid_m[base + I] += weight * p_mass


@ti.func
def in_bound(index, offset, shape):
    res = True
    for i in ti.static(range(len(shape))):
        if index[i] + offset[i] < 0 or index[i] + offset[i] >= shape[i]:
            res = False
    return res


@ti.kernel
def p2g(x: ti.template(), dx: ti.f32, grid_m: ti.template(), dim: ti.template()):
    """
    将周围粒子的质量scatter到网格上。实际上，只要替换grid_m, 可以scatter任何标量场。
    (此函数相比p2g_2d与pg2_3d来说，增加了参数dim)
    Args:
        x (ti.template()): 粒子位置
        dx (ti.f32): 网格间距
        grid_m (ti.template()): 网格质量(输出)
        dim (ti.template()): 网格维度
    """
    inv_dx = 1.0 / dx
    p_mass = 1.0
    for p in x:
        base = ti.cast(ti.floor(x[p] * inv_dx - 0.5), ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # Loop over 3x3 grid node neighborhood
        for offset in ti.static(ti.grouped(ti.ndrange(*((3,) * dim)))):
            weight = 1.0
            for d in ti.static(range(dim)):
                weight *= w[offset[d]][d]
            if in_bound(base, offset, grid_m.shape):
                grid_m[base + offset] += weight * p_mass



############################################
# debug info

            

def debug_info(field, name="", dont_print_cli=False):
    field_np = field.to_numpy()
    if name == "":
        name = field._name
    print("---------------------")
    print("name: ", name)
    print("shape: ", field_np.shape)
    print("min, max: ", field_np.min(), field_np.max())
    if not dont_print_cli:
        print(field_np)
    print("---------------------")
    if field_np.ndim > 2:
        np.savetxt(f"result/debug_{name}.csv", field_np.reshape(-1, field_np.shape[-1]), fmt="%.2f", delimiter="\t")
    else:
        np.savetxt(f"result/debug_{name}.csv", field_np, fmt="%.2f", delimiter="\t")
    return field_np





############################################
# collider
# from taichi.math import vec2, vec3, dot, clamp, length, sign, sqrt, min, max

# ref: https://iquilezles.org/articles/distfunctions/
@ti.func
def sphere(pos, radius):
    return pos.norm() - radius


@ti.func
def box(p, b):
    q = abs(p) - b
    return ti.math.length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0)


@ti.func
def torus(p, t: ti.math.vec2):
    q = vec2(ti.math.length(p.xz) - t.x, p.y)
    return ti.math.length(q) - t.y


@ti.func
def plane(pos, normal, height):
    return pos.dot(normal) + height


@ti.func
def cone(p, c, h):
    q = h * vec2(c.x / c.y, -1.0)
    w = vec2(ti.math.length(p.xz), p.y)
    a = w - q * ti.math.clamp(ti.math.dot(w, q) / ti.math.dot(q, q), 0.0, 1.0)
    b = w - q * vec2(ti.math.clamp(w.x / q.x, 0.0, 1.0), 1.0)
    k = ti.math.sign(q.y)
    d = min(ti.math.dot(a, a), ti.math.dot(b, b))
    s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y))
    return ti.math.sqrt(d) * ti.math.sign(s)


@ti.func
def union(a, b):
    return min(a, b)


@ti.func
def intersection(a, b):
    return max(a, b)


@ti.func
def subtraction(a, b):
    return max(-a, b)


@ti.func
def triangle(p: ti.math.vec3, a: ti.math.vec3, b: ti.math.vec3, c: ti.math.vec3):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
    nor = ba.cross(ac)

    res = nor.dot(pa) * nor.dot(pa) / nor.norm_sqr()
    if ti.math.sqrt(ti.math.sign(ba.cross(nor).dot(pa)) + ti.math.sign(cb.cross(nor).dot(pb)) + ti.math.sign(ac.cross(nor).dot(pc))) < 2.0:
        res = min(
            min(
                (ba * ti.math.clamp(ba.dot(pa) / ba.dot(ba), 0.0, 1.0) - pa).norm(),
                (cb * ti.math.clamp(cb.dot(pb) / cb.dot(cb), 0.0, 1.0) - pb).norm(),
            ),
            (ac * ti.math.clamp(ac.dot(pc) / ac.dot(ac), 0.0, 1.0) - pc).norm(),
        )
    return res


@ti.func
def collision_response_sdf(pos: ti.template(), sdf):
    sdf_epsilon = 1e-4
    grid_idx = ti.Vector([pos.x * sdf.resolution, pos.y * sdf.resolution, pos.z * sdf.resolution], ti.i32)
    grid_idx = ti.math.clamp(grid_idx, 0, sdf.resolution - 1)
    normal = sdf.grad[grid_idx]
    sdf_val = sdf.val[grid_idx]
    assert 1 - 1e-4 < normal.norm() < 1 + 1e-4, f"sdf normal norm is not one: {normal.norm()}"
    if sdf_val < sdf_epsilon:
        pos -= sdf_val * normal



def csr_is_equal(A, B):
    if A.shape != B.shape:
        print("shape not equal")
        return False
    diff = A - B
    if diff.nnz == 0:
        return True
    maxdiff = np.abs(diff.data).max()
    print("maxdiff: ", maxdiff)
    if maxdiff > 1e-6:
        return False
    return True
