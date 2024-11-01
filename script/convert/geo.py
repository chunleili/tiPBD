""" Convert Houdini .geo file (a ascii json)"""


import json
import os
import sys
import numpy as np
import argparse

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description='Convert VTK to PLY')
parser.add_argument('-input', type=str, help='Input VTK file')
args = parser.parse_args()

from dataclasses import dataclass


# refered python script https://github.com/cgdougm/HoudiniObj/blob/master/hgeo.py
# Houdini geo format https://www.sidefx.com/docs/houdini/io/formats/geo.html
# Houdini primitive format https://www.sidefx.com/docs/houdini/model/primitives.html


# Polygon: "Poly"
# NURBS Curve: "NURBCurve"
# Rational Bezier Curve: "BezierCurve"
# Linear Patch: "Mesh"
# NURBS Surface: "NURBMesh"
# Rational Bezier Patch: "BezierMesh"
# Ellipse/Circle: "Circle"
# Ellipsoid/Sphere: "Sphere"
# Tube/Cone: "Tube" Metaball "MetaBall"
# Meta Super-Quadric: "MetaSQuad"
# Particle System:  "Part"
# Paste Hierarchy: "PasteSurf"
PrimitiveType = {
"Poly": 0,
"NURBCurve": 1,
"BezierCurve": 2,
"Mesh": 3,
"NURBMesh": 4,
"BezierMesh": 5,
"Circle": 6,
"Sphere": 7,
"Tube": 8,
"MetaBall": 9,
"MetaSQuad": 10,
"Part": 11,
"PasteSurf": 12,
}




@dataclass
class PrimAttr:
    name:str = None
    size:int = None
    dtype:str = None
    value = None
    def __init__(self, name, size, dtype, value):
        self.name = name
        self.size = size
        self.dtype = dtype
        self.value = value

class Geo:
    """
    .geo data has 9 attributes：
    1. fileversion
    2. hasindex
    3. pointcount
    4. vertexcount
    5. primitivecount
    6. info
    7. topology
        7.1 pointref 
           7.1.1 inidices
    8. attributes
        8.1 pointattributes
        8.2 primitiveattributes
    9. primitives
    """

    def __init__(self, input:str=None):
        if input:
            self.read(input)

    
    @staticmethod
    def _pairListToDict(pairs):
        return dict( zip(pairs[0::2],pairs[1::2]) )

    # def read(self, input:str):
    #     self.input = input
    #     with open(input, "r") as f:
    #         self.raw = json.load(f)
    #     # 读取顶点个数等信息
    #     self.pointcount = self.raw[5]
    #     self.vertexcount = self.raw[7]
    #     self.primitivecount = self.raw[9]
    #     # 读取顶点索引
    #     self.topology = self.raw[13]
    #     self.pointref = self.topology[1]
    #     self.indices = self.pointref[1] # IMPORTANT
    #     # 读取顶点的位置
    #     self.attributes = self.raw[15]
    #     self.pointattributes = self.attributes[1]
    #     self.primitiveattributes = self.attributes[3]
    #     self.positions = self.pointattributes[0][1][7][5] # IMPORTANT
    #     self._extract_prim()
    #     print("Finish reading geo file: ", input)


    def read(self,filePath):
        with open(filePath, 'r') as fp:
            self.raw = json.load(fp)
    
        for name,item in zip(self.raw[0::2],self.raw[1::2]):
            self.__setattr__(name,item)

        self.topology = self._pairListToDict(self.topology)
        self.pointref = self._pairListToDict(self.topology['pointref'])

        self.attributes = self._pairListToDict(self.attributes)
        print("Finish reading geo file: ", filePath)
    
    @staticmethod
    def _pairListToDict(pairs):
        return dict( zip(pairs[0::2],pairs[1::2]) )


    def write(self, output:str=None):
        if output:
            self.output = output
        else:
            self.output = str(Path(self.input).parent) + "/" + str(Path(self.input).stem) + ".geo"

        with open(self.output, "w") as f:
            json.dump(self.rawgeo, f)
        print("Finish writing geo file: ", self.output)


    def set_positions(self,positions):
        self.positions = positions
        self.pointcount = len(positions)

    # trianlge version
    # TODO: add other primitive types
    def read_vtk(self,input:str):
        import meshio
        self.input = input
        mesh = meshio.read(input, file_format="vtk")
        self.pointcount = len(mesh.points)
        self.positions = mesh.points
        tri = mesh.cells_dict['triangle']
        self.strain = mesh.cell_data['strain'][0] # 读取应变信息
        self.primitivecount = len(tri)
        self.indices = tri.flatten()

    
class Polygon(object):
    def __init__(self,indices,closed=False):
        self.indicies = indices
        self.closed   = closed

def read_geo(input):
    geo = Geo(input)
    return geo


if __name__ == '__main__':
    from pathlib import Path
    dir = str(Path(__file__).parent) + "/"
    geo = read_geo(dir+"sample_in.geo")
    geo.read_vtk(dir+"sample.vtk")
    geo.write("sample_out.geo")