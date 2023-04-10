from engine.sdf import SDF
def test_sdf_2d():
    sdf = SDF((5, 5))
    sdf.val.fill(1)
    print(sdf.val)
    print(sdf.grad)
    sdf.compute_gradient(1.0,1.0)  
    print(sdf)
    sdf.print_to_file()    

def test_sdf_3d():
    sdf_3d = SDF((5, 5, 5))
    sdf_3d.val.fill(1)
    print(sdf_3d.val)
    print(sdf_3d.grad)
    sdf_3d.compute_gradient(1.0,1.0,1.0)
    print(sdf_3d)
    sdf_3d.print_to_file("result/sdf_3d")

if __name__ == "__main__":
    test_sdf_2d()
    test_sdf_3d()