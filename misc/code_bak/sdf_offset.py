import taichi as ti

ti.init(debug=True)


@ti.data_oriented
class SDF:
    def __init__(self, shape, dx=1.0, dy=1.0, dz=1.0):
        self.dim = len(shape)
        self.shape = shape
        print("SDF init...")
        if self.dim == 2:
            self.new_shape = (shape[0] + 2, shape[1] + 2)
            self.val = ti.field(dtype=ti.f32, shape=self.new_shape, offset=(-1, -1))
            self.grad = ti.Vector.field(self.dim, dtype=ti.f32, shape=shape)
            self.compute_gradient_2d(dx, dy)
        elif self.dim == 3:
            self.new_shape = (shape[0] + 2, shape[1] + 2, shape[2] + 2)
            self.val = ti.field(dtype=ti.f32, shape=self.new_shape, offset=(-1, -1, -1))
            self.grad = ti.Vector.field(self.dim, dtype=ti.f32, shape=shape)
            self.compute_gradient_3d(dx, dy, dz)
        else:
            raise Exception("SDF only supports 2D/3D for now")

        self.compute_gradient(dx, dy, dz)

    def compute_gradient(self, dx, dy, dz=None):
        """
        Compute the gradient of the SDF field.
        """
        if self.dim == 2:
            self.compute_gradient_2d(dx, dy)
        elif self.dim == 3:
            self.compute_gradient_3d(dx, dy, dz)
        else:
            raise Exception("SDF only supports 2D/3D for now")

    @ti.kernel
    def compute_gradient_2d(self: ti.template(), dx: ti.f32, dy: ti.f32):
        for i, j in self.grad:
            self.grad[i, j] = ti.Vector(
                [(self.val[i + 1, j] - self.val[i - 1, j] / dx), (self.val[i, j + 1] - self.val[i, j - 1]) / dy]
            )

    @ti.kernel
    def compute_gradient_3d(self: ti.template(), dx: ti.f32, dy: ti.f32, dz: ti.f32):
        for i, j, k in self.grad:
            self.grad[i, j, k] = ti.Vector(
                [
                    (self.val[i + 1, j, k] - self.val[i - 1, j, k] / dx),
                    (self.val[i, j + 1, k] - self.val[i, j - 1, k]) / dy,
                    (self.val[i, j, k + 1] - self.val[i, j, k - 1]) / dz,
                ]
            )

    def to_numpy(self):
        return self.val.to_numpy(), self.grad.to_numpy()

    def __str__(self) -> str:
        return "shape:\n" + str(self.shape) + "\n\nval:\n" + str(self.val) + "\n\n" + "grad:\n" + str(self.grad)

    def print_to_file(self, filename="sdf"):
        import numpy as np

        val, grad = self.to_numpy()
        np.savetxt(filename + "_val.txt", val)
        np.savetxt(filename + "_grad.txt", grad)


if __name__ == "__main__":
    sdf = SDF((3, 3))
    sdf.val.fill(1)
    sdf.compute_gradient(1.0, 1.0)
    # print(sdf.val)
    # print(sdf)
