import taichi as ti

@ti.data_oriented
class SDF:
    '''
    Signed Distance Field (SDF) class.

    This implementation tackles the boundary issue with if statements in the for loops. This will preserve the sparsity of the field, but will cause if statements in the for loops.
    '''
    def __init__(self, shape):
        print("SDF init...")
        # self.dim = len(shape)
        # self.shape = shape
        # self.val =  ti.field(dtype=ti.f32, shape=shape)
        # self.grad = ti.Vector.field(self.dim, dtype=ti.f32, shape=shape)

    def generate_from_mesh(self, mesh_path):
        '''
        Generate SDF from mesh file.
        '''
        import trimesh, mesh_to_sdf
        mesh = trimesh.load(mesh_path)
        mesh = mesh_to_sdf.scale_to_unit_cube(mesh)
        sdf = mesh_to_sdf.sample_sdf_near_surface(mesh,number_of_points=10000)
        sdf_val, sdf_grad = sdf[0], sdf[1]
        self.shape = sdf_val.shape
        self.dim = len(self.shape)
        self.val = ti.field(dtype=ti.f32, shape=self.shape)
        self.grad = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.shape)
        self.val.from_numpy(sdf_val)
        self.grad.from_numpy(sdf_grad)


    def compute_gradient(self, dx, dy, dz=None):
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
     

    def to_numpy(self):
        return self.val.to_numpy(), self.grad.to_numpy()

    def __str__(self) -> str:
         return "shape:\n"+str(self.shape)+"\n\nval:\n" + str(self.val) + "\n\n" + "grad:\n" + str(self.grad)
    
    def print_to_file(self, filename="result/sdf"):
        import numpy as np
        val, grad = self.to_numpy()
        if self.dim == 2:
            np.savetxt(filename+"_val.txt", val, fmt="%.2e")
            np.savetxt(filename+"_grad.txt", grad.reshape(-1, self.dim), fmt="%.2e")
        elif self.dim == 3:
            np.savetxt(filename+"_val.txt", val.flatten(), fmt="%.2e")
            np.savetxt(filename+"_grad.txt", grad.reshape(-1, self.dim), fmt="%.2e")


class SDFBase:
    def __init__(self, shape):
        print("SDF init...\nresolution: "+str(shape)+"...")
        self.dim = len(shape)
        self.shape = shape
        self.val =  ti.field(dtype=ti.f32, shape=shape)
        self.grad = ti.Vector.field(self.dim, dtype=ti.f32, shape=shape)


def mesh2sdf(mesh_path='data/model/chair.obj', scale_to_unit_cube=True):
    import trimesh, mesh_to_sdf
    mesh = trimesh.load(mesh_path)
    mesh = mesh_to_sdf.scale_to_unit_cube(mesh)
    sdf = mesh_to_sdf.sample_sdf_near_surface(mesh,number_of_points=10000)
    sdf_val, sdf_grad = sdf[0], sdf[1]
    return sdf_val, sdf_grad


# def gen_sdf_p2g(shape, mesh_path):
#     '''
#     从表面网格生成SDF场。
#     '''
#     import trimesh
#     from mesh_io import scale_to_unit_cube, shift
#     from p2g import p2g
 
#     mesh = trimesh.load(mesh_path)
#     mesh = scale_to_unit_cube(mesh)
#     shift(mesh, (1.0, 1.0, 1.0))
#     x = ti.Vector.field(3, dtype=ti.f32, shape=mesh.vertices.shape[0])
#     x.from_numpy(mesh.vertices)
#     grid_m = ti.field(dtype=ti.f32, shape=shape)
#     p2g(x, 0.1, grid_m, 3)
#     return grid_m


def gen_sdf_voxel(mesh_path='data/model/chair.obj', resolution=64):
    '''
    从表面网格生成SDF场。借助mesh_to_sdf库和trimesh。

    Args:
        mesh_path (str, optional): 网格文件路径。 Defaults to 'data/model/chair.obj'.
        resolution (int, optional): 体素分辨率。 Defaults to 64.
    '''
    import trimesh, mesh_to_sdf
    mesh = trimesh.load(mesh_path)
    mesh = mesh_to_sdf.scale_to_unit_cube(mesh)
    vox = mesh_to_sdf.mesh_to_voxels(mesh, voxel_resolution=resolution)
    return vox
# ---------------------------------------------------------------------------- #
#                                     test                                     #
# ---------------------------------------------------------------------------- #

def test_sdf_basic():
    sdf = SDF((5, 5))
    sdf.val.fill(1)
    print(sdf.val)
    print(sdf.grad)
    sdf.compute_gradient(1.0,1.0)  
    print(sdf)
    sdf.print_to_file()    

    sdf_3d = SDF((5, 5, 5))
    sdf_3d.val.fill(1)
    print(sdf_3d.val)
    print(sdf_3d.grad)
    sdf_3d.compute_gradient(1.0,1.0,1.0)
    print(sdf_3d)
    sdf_3d.print_to_file("result/sdf_3d")

def test_mesh2sdf():
    sdf_val, sdf_grad = mesh2sdf()
    pass
    from visualize import visualize
    visualize(sdf_val)
    # print(sdf_val)
    # print(sdf_grad)
    
def test_sdf_from_mesh():
    sdf = SDF((5, 5))
    sdf.from_mesh('data/model/chair.obj')
    print(sdf)
    sdf.print_to_file("result/sdf_from_mesh")


def test_gen_sdf_voxel():
    vox = gen_sdf_voxel('data/model/chair.obj',64)
    from visualize import vis_grid
    vis_grid(vox)

if __name__ == "__main__":
    ti.init(arch=ti.cuda)

    test_gen_sdf_voxel()
    # test_mesh2voxels()
    