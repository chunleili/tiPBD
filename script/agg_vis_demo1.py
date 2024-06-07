# 2D example of viewing aggregates from SA using VTK
# https://github.com/pyamg/pyamg-examples?tab=readme-ov-file#visualizingaggregation
import pyamg
import pyamg.vis

case_name = "scale12"

import pathlib
# mkdir if not exists
pathlib.Path(f'result/vis').mkdir(parents=True, exist_ok=True)

def vis_aggs(frame=10):
    import meshio
    mesh_path = f"result/{case_name}/obj/{frame:04d}.obj"
    mesh = meshio.read(mesh_path)
    V = mesh.points
    E2V = mesh.cells_dict['triangle']
    # mesh = trimesh.load(mesh_path)
    # V = mesh.vertices
    # E2V = mesh.faces

    from scipy.io import  mmread
    A_path = f"result/{case_name}/A/A_F{frame:d}-0.mtx"
    A = mmread(A_path).tocsr()

    # perform smoothed aggregation
    AggOp, rootnodes = pyamg.aggregation.standard_aggregation(A)

    # create the vtk file of aggregates
    pyamg.vis.vis_coarse.vis_aggregate_groups(V=V, E2V=E2V, AggOp=AggOp,
                                            mesh_type='tri', fname=f'result/vis/{case_name}_aggs_{frame}.vtu')

    # create the vtk file for a mesh
    pyamg.vis.vtk_writer.write_basic_mesh(V=V, E2V=E2V,
                                        mesh_type='tri', fname=f'result/vis/{case_name}_mesh_{frame}.vtu')


for frame in range(1, 30):
    vis_aggs(frame)











# retrieve the problem
# data = pyamg.gallery.load_example('unit_square')
# A = data['A'].tocsr()
# V = data['vertices']
# E2V = data['elements']

use_vedo = False
if use_vedo:
    import vedo
    gmesh = vedo.load('output_mesh.vtu')
    gaggs = vedo.load('output_aggs.vtu')

    gmesh = gmesh.tomesh().color('w').alpha(0.1)
    gmesh.color('gray')
    gmesh.lw(3.0)

    agg3 = []
    agg2 = []
    for cell in gaggs.cells:
        if len(cell) == 2:
            agg2.append(cell)
        else:
            agg3.append(cell)

    mesh2 = vedo.Mesh([gaggs.points(), agg2])
    mesh3 = vedo.Mesh([gaggs.points(), agg3])
    mesh2.linecolor('blue').linewidth(8)
    mesh3.color('r').linewidth(0)

    figname = './vis_aggs2.png'
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--savefig':
            plt = vedo.Plotter(offscreen=True)
            plt += gmesh
            plt += mesh2
            plt += mesh3
            plt.show().screenshot(figname)
    else:
        plt = vedo.Plotter()
        plt += gmesh
        plt += mesh2
        plt += mesh3
        plt.show()


# to use Paraview:
# start Paraview: Paraview --data=output_mesh.vtu
# apply
# under display in the object inspector:
#           select wireframe representation
#           select a better solid color
# open file: output_aggs.vtu
# under display in the object inspector:
#           select surface with edges representation
#           select a better solid color
#           increase line width and point size to see these aggs (if present)