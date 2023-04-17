import taichi as ti
import os
import numpy as np

@ti.data_oriented
class SDF:
    '''
    Signed Distance Field (SDF) class.
    '''
    def __init__(self, mesh_path=None, resolution=64, dim=3):
        '''
        生成SDF体素场。其中有两个taichi field: val和grad，分别表示SDF体素场的值和梯度。

        Args:
            mesh_path (str): 网格文件路径，如果为None则需要手动填充SDF体素场。
            resolution (int): SDF体素场分辨率。默认为64。
            dim (int): SDF体素场维度。默认为3。
        '''
        self.resolution = resolution
        if dim==2:
            self.shape = (resolution, resolution)
        elif dim==3:
            self.shape = (resolution, resolution, resolution)
        else:
            raise Exception("SDF only supports 2D/3D for now")
        self.dim = dim
        self.val =  ti.field(dtype=ti.f32, shape=self.shape)
        self.grad = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.shape)

        print("SDF resolution = ", self.shape)
        print("SDF initilizing...")

        if mesh_path is not None:
            print(mesh_path)
            # 检查是否存在cache
            val_cache_path = mesh_path+ "_sdf_cache_val.npy"
            grad_cache_path = mesh_path+ "_sdf_cache_grad.npy"
            if not os.path.exists(val_cache_path) or not os.path.exists(grad_cache_path):
                print("No sdf cache found. Generating sdf cache...")
                val_np, grad_np = gen_sdf_voxels(mesh_path, resolution, True)
                np.save(val_cache_path, val_np)
                np.save(grad_cache_path,grad_np)
            else:
                print("Found sdf cache. Loading sdf cache...")
                val_np = np.load(val_cache_path)
                grad_np = np.load(grad_cache_path)

            self.val.from_numpy(val_np)
            self.grad.from_numpy(grad_np)
        
        print("SDF init done.")


    def compute_gradient(self, dx=1.0, dy=1.0, dz=1.0):
        '''
        Compute the gradient of the SDF field.
        '''
        if self.dim == 2:
            self.compute_gradient_2d(dx, dy)
        elif self.dim == 3:
            self.compute_gradient_3d(dx, dy, dz)
        else:
            raise Exception("SDF only supports 2D/3D for now")

    @ti.kernel
    def compute_gradient_2d(self, dx: ti.f32, dy: ti.f32):
        for i, j in self.grad:
            if 0<i<self.shape[0]-1 and 0<j<self.shape[1]-1:
                self.grad[i, j] = ti.Vector([(self.val[i+1, j] - self.val[i-1, j]/dx), (self.val[i, j+1] - self.val[i, j-1])/dy]) * 0.5 


    @ti.kernel
    def compute_gradient_3d(self, dx: ti.f32, dy: ti.f32, dz: ti.f32):
        for i, j, k in self.grad:
            if 0<i<self.shape[0]-1 and 0<j<self.shape[1]-1 and 0<k<self.shape[2]-1:
                self.grad[i, j, k] = ti.Vector([
                    (self.val[i+1, j, k] - self.val[i-1, j, k])/dx,
                    (self.val[i, j+1, k] - self.val[i, j-1, k])/dy,
                    (self.val[i, j, k+1] - self.val[i, j, k-1])/dz
                    ]) * 0.5

    def __str__(self) -> str:
         return "shape:\n"+str(self.shape)+"\n\nval:\n" + str(self.val) + "\n\n" + "grad:\n" + str(self.grad)    


class SDFBase:
    def __init__(self, shape):
        print("SDF init...\nresolution: "+str(shape)+"...")
        self.dim = len(shape)
        self.shape = shape
        self.val =  ti.field(dtype=ti.f32, shape=shape)
        self.grad = ti.Vector.field(self.dim, dtype=ti.f32, shape=shape)


def gen_sdf_points(mesh_path):
    '''
    从表面网格生成采样点(靠近网格表面处的)SDF场。借助mesh_to_sdf库和trimesh。注意导入的模型会自动缩放到[-1,1]的立方体内。
    '''
    import trimesh, mesh_to_sdf
    mesh = trimesh.load(mesh_path)
    mesh = mesh_to_sdf.scale_to_unit_cube(mesh)
    sdf = mesh_to_sdf.sample_sdf_near_surface(mesh,number_of_points=10000)
    sdf_val, sdf_grad = sdf[0], sdf[1]
    return sdf_val, sdf_grad


def gen_sdf_voxels(mesh_path, resolution=64, return_gradients=False):
    '''
    从表面网格生成体素(靠近网格表面处的)SDF场。借助mesh_to_sdf库和trimesh。注意导入的模型会自动缩放到[-1,1]的立方体内。

    Args:
        mesh_path (str, optional): 网格文件路径。 Defaults to 'data/model/chair.obj'.
        resolution (int, optional): 体素分辨率。 Defaults to 64.
    '''
    import trimesh, mesh_to_sdf
    from mesh_io import scale, shift
    from engine.metadata import meta
    mesh = trimesh.load(mesh_path)
    mesh = mesh_to_sdf.scale_to_unit_cube(mesh)
    if meta.get_common("sdf_mesh_scale") is not None:
        scale(mesh, meta.get_common("sdf_mesh_scale"))
    if meta.get_common("sdf_mesh_shift") is not None:
        shift(mesh, meta.get_common("sdf_mesh_shift"))
    vox = mesh_to_sdf.mesh_to_voxels(mesh, voxel_resolution=resolution, return_gradients=return_gradients)
    return vox