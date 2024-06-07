# 3D example of viewing aggregates from SA using VTK
# https://github.com/pyamg/pyamg-examples?tab=readme-ov-file#visualizingaggregation
import pyamg
import pyamg.vis

# retrieve the problem
data = pyamg.gallery.load_example('unit_cube')
A = data['A'].tocsr()
V = data['vertices']
E2V = data['elements']

# perform smoothed aggregation
AggOp, rootnodes = pyamg.aggregation.standard_aggregation(A)

# create the vtk file of aggregates
pyamg.vis.vis_coarse.vis_aggregate_groups(V=V, E2V=E2V, AggOp=AggOp,
                                          mesh_type='tet', fname='output_aggs.vtu')

# create the vtk file for a mesh
pyamg.vis.vtk_writer.write_basic_mesh(V=V, E2V=E2V,
                                      mesh_type='tet', fname='output_mesh.vtu')

# try:
import vedo
gmesh = vedo.load('output_mesh.vtu')
gaggs = vedo.load('output_aggs.vtu')

gmesh = gmesh.tomesh().color('w').alpha(0.1)
gmesh.color('gray')
gmesh.lw(3.0)

agg2 = []
agg3 = []
agg4 = []
for cell in gaggs.cells:
    if len(cell) == 2:
        agg2.append(cell)
    elif len(cell) == 3:
        agg3.append(cell)
    else:
        agg4.append(cell)

mesh2 = vedo.Mesh([gaggs.points(), agg2])
mesh3 = vedo.Mesh([gaggs.points(), agg3])
mesh4 = vedo.Mesh([gaggs.points(), agg4])
mesh2.linecolor('b').linewidth(8)
mesh3.color('b').linewidth(0)
mesh4.color('b').linewidth(0)

figname = './output/vis_aggs3.png'
import sys
if len(sys.argv) > 1:
    if sys.argv[1] == '--savefig':
        plt = vedo.Plotter(offscreen=True)
        plt += gmesh
        plt += mesh2
        plt += mesh3
        plt += mesh4
        plt.show(camera={'pos': (3, 3, 3)}).screenshot(figname)
else:
    plt = vedo.Plotter()
    plt += gmesh
    plt += mesh2
    plt += mesh3
    plt += mesh4
    plt.show(camera={'pos': (3, 3, 3)})
# except:
#     pass
#     print("Failed to visualize aggregates using vedo. Please install vedo.")

# to use Paraview:
# start Paraview: Paraview --data=output_mesh.vtu
# apply
# under display in the object inspector:
#           select wireframe representation
#           select a better solid color
#           selecting surface with edges and low opacity also helps
# open file: output_aggs.vtu
# under display in the object inspector:
#           select surface with edges representation
#           select a better solid color
#           increase line width and point size to see these aggs (if present)
#           reduce the opacity, sometimes helps