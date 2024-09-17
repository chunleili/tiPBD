import trimesh
import numpy as np
import pathlib
import shutil
import argparse

result_dir = pathlib.Path(__file__).resolve().parent.parent / "result"

parser = argparse.ArgumentParser()
parser.add_argument("-case_name", default= "scale64-cpu")
parser.add_argument("-start_frame", default=0)
parser.add_argument("-end_frame", default=50)
parser.add_argument("-use_pyplot", default=True)
parser.add_argument("-use_trimesh", default=False)
parser.add_argument("-inspect", type=int, default=-1, help="Inspect the specific frame by trimesh. Default is -1 (no inspection). If set, start_frame and end_frame will be ignored.")

args = parser.parse_args()
case_name = args.case_name
start_frame = args.start_frame
end_frame = args.end_frame
use_pyplot = args.use_pyplot
use_trimesh = args.use_trimesh
inspect = args.inspect

dir = result_dir / f"{case_name}" / "obj"

print(f"visualize obj files (frame {start_frame} to {end_frame}) in {dir}")
print("For trimesh: press w to wireframe mode")

if inspect >= 0:
    start_frame = inspect
    end_frame = inspect+1
    use_pyplot = False
    use_trimesh = True


if use_pyplot:
    import matplotlib.pyplot as plt
    # fig = plt.figure()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')


def render_by_trimesh(mesh_):
    mesh = mesh_.copy()
    # set the pinned vertex color to red
    mesh.visual.vertex_colors[0] = (255, 0, 0, 255) 
    mesh.visual.vertex_colors[63] = (255, 0, 0, 255)
    # roatate the mesh to see it. Because trimesh only render one-side
    angle, axis = -np.pi/2, [1,0, 0]
    angle2, axis2 = -np.pi/2, [0,0, 1]
    t1 = trimesh.transformations.rotation_matrix(angle, axis)
    t2 = trimesh.transformations.rotation_matrix(angle2, axis2)
    t = t2@t1
    mesh.apply_transform(t)
    scene = trimesh.Scene([mesh])
    # print(scene.camera_transform)
    scene.show(smooth=False)

def render_by_pyplot(mesh):
    ax.clear()
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)
    ax.set_title(f"frame {i}")
    plt.pause(0.2)

# load all obj files
meshes = []
for i in range(start_frame, end_frame):
    obj_file = dir / f"{i}.obj"
    if not obj_file.exists():
        if pathlib.Path(dir / f"{i:04d}.obj").exists():
            obj_file = dir / f"{i:04d}.obj"
        else:
            print(f"{obj_file} not found.")
            continue
    print(f"Loading {obj_file.name}...")
    # load by trimesh
    mesh = trimesh.load_mesh(obj_file)
    meshes.append(mesh)

# render all obj files
for loop in range(10):
    for i, mesh in enumerate(meshes):
        print(f"Rendering frame {i}")
        if use_trimesh:
            render_by_trimesh(mesh)
        if use_pyplot:
            render_by_pyplot(mesh)  
    inp = input("Do you want to render again? (y/Y to continue)")
    if inp != "y" and inp != "Y":
        break
# render_by_pyplot(mesh) 
