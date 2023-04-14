import taichi as ti

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
        print("SDF init...")
        self.resolution = resolution
        if dim==2:
            self.shape = (resolution, resolution)
        elif dim==3:
            self.shape = (resolution, resolution, resolution)
        else:
            raise Exception("SDF only supports 2D/3D for now")
        print("SDF resolution = ", self.shape)
        self.dim = dim
        self.val =  ti.field(dtype=ti.f32, shape=self.shape)
        self.grad = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.shape)

        if mesh_path is not None:
            val_np, grad_np = gen_sdf_voxels(mesh_path, resolution, True)
            self.val.from_numpy(val_np)
            self.grad.from_numpy(grad_np)


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
    mesh = trimesh.load(mesh_path)
    mesh = mesh_to_sdf.scale_to_unit_cube(mesh)
    vox = mesh_to_sdf.mesh_to_voxels(mesh, voxel_resolution=resolution, return_gradients=return_gradients)
    return vox
# ---------------------------------------------------------------------------- #
#                                     test                                     #
# ---------------------------------------------------------------------------- #
def test_sdf_basic():
    # fill with 1
    sdf = SDF(None, 5,dim=2)
    sdf.val.fill(1)
    sdf.compute_gradient(1.0,1.0)  
    print(sdf)

    # fill with 1, 3d
    sdf_3d = SDF(None, 5,dim=3)
    sdf_3d.val.fill(1)
    sdf_3d.compute_gradient(1.0,1.0,1.0)
    print(sdf_3d)

    #random fill, 3d
    import numpy as np
    sdf_3d = SDF(None, 5,dim=3)
    sdf_3d.val.from_numpy(np.random.rand(5,5,5))
    sdf_3d.compute_gradient()
    print(sdf_3d)

def test_gen_sdf_points():
    val, grad = gen_sdf_points()
    from visualize import visualize
    visualize(val)
    
def test_gen_sdf_voxels():
    vox, grad = gen_sdf_voxels('data/model/chair.obj',64,True)
    from visualize import vis_sdf
    vis_sdf(vox)

def test_SDF():
    sdf = SDF('data/model/chair.obj',64, 3)
    from visualize import vis_sdf
    vis_sdf(sdf.val)

if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    # test_sdf_basic()
    # test_gen_sdf_voxels()
    test_SDF()