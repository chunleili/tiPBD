""" Convert Houdini .geo file (a ascii json)"""


import json
import os
import sys
import numpy as np
import argparse
from pathlib import Path

sys.path.append(os.getcwd())

def get_args():
    parser = argparse.ArgumentParser(description='Convert VTK to PLY')
    parser.add_argument('-input', type=str, default="", help='Input VTK file')
    args = parser.parse_args()
    return args

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


    def __init__(self, input:str=None, only_P=False):
        self.only_P = only_P
        if input:
            self.input = input
            self.read(input)

    
    @staticmethod
    def _pairListToDict(pairs):
        return dict( zip(pairs[0::2],pairs[1::2]) )

    def read(self,filePath):
        with open(filePath, 'r') as fp:
            self.raw = json.load(fp)
    
        for name,item in zip(self.raw[0::2],self.raw[1::2]):
            self.__setattr__(name,item)

        self.topology = self._pairListToDict(self.topology)
        self.pointref = self._pairListToDict(self.topology['pointref'])

        self.attributes = self._pairListToDict(self.attributes)

        if self.only_P:
            self.parse_pointattributes()
            print("Finish reading geo file: ", filePath)
            return
        
        self.parse_vert()
        self.parse_pointattributes()
        self.parse_primattributes()

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
            json.dump(self.raw, f)
        print("Finish writing geo file: ", self.output)


    def set_positions(self,pos):
        if isinstance(pos, np.ndarray):
            pos = pos.tolist()
        self.positions[:] = pos
        self.pointcount = len(pos)

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

    def get_gluetoaniamtion(self):
        if not hasattr(self, 'gluetoanimation'):
            self.gluetoanimation = [0]*self.pointcount
        return self.gluetoanimation
    
    def get_pin(self):
        self.pin = self.gluetoanimation
        return self.pin
    
    def parse_vert(self):
        if 'indices' not in self.pointref:
            return None
        if self.primitivecount==0:
            return None
        self.indices = self.pointref['indices']
        self.NVERT_ONE_CONS = len(self.indices)//self.primitivecount
        self.NCONS = self.primitivecount
        self.vert = np.array(self.indices).reshape(self.NCONS, self.NVERT_ONE_CONS).tolist()
        return self.vert
    
    def get_vert(self):
        return self.vert

    def parse_pointattributes(self):
        self.pointattributes = self.attributes['pointattributes']
        class AttributeValue:
            None
        class PointAttr:
            None
        # parse point attributes
        allPointAttr = []
        for i in range(len(self.pointattributes)):
            attrRaw0 = self.pointattributes[i][0] #metadata
            a = PointAttr()
            for name,item in zip(attrRaw0[0::2],attrRaw0[1::2]):
                a.__setattr__(name,item)

            attrRaw1 = self.pointattributes[i][1] #data
            for name,item in zip(attrRaw1[0::2],attrRaw1[1::2]):
                a.__setattr__(name,item)

            if a.name == "P":
                self.positions = a.values[5]
            if a.name == "gluetoanimation":
                self.gluetoanimation = a.values[5][0]
            if a.name == "mass":
                self.mass = a.values[5][0]
            if a.name == "pts":
                self.pts = a.values[5][0]
            if a.name == "target_pos":
                self.target_pos = a.values[5]
            allPointAttr.append(a)
        return self.positions

    def get_pos(self):
        return self.positions
    

    def get_extraSpring(self):
        if  hasattr(self, 'extraSpring'):
            return self.extraSpring
        elif hasattr(self, 'target_pt') and hasattr(self, 'pts'):
            self.parse_extraSpring_from_target_pt()
            return self.extraSpring
        else:
            raise Exception("No extraSpring or target_pt and pts found in geo file")


    def parse_extraSpring_from_dict(self,extraSpring):
        """
        将list of dict数据的extraSpring转换为vert
        第一个点为拉动点(bone_pt_idex)，第二个点为被拉动点(muscle_pt_idex)
        """
        extraSpring_vert = []
        for i in range(len(extraSpring)):
            extraSpring_vert.append([extraSpring[i]['bone_pt_index']['value'],extraSpring[i]['bone_pt_index']['value']])
        self.extraSpring = extraSpring_vert
        ...


    def parse_extraSpring_from_target_pt(self):
        """
        给定target_pt和pts，生成extraSpring(vert格式)
        target_pt为拉动点(driving point)，pts为被拉动点(to be driven point)
        """
        extraSpring_vert = []
        for i in range(len(self.target_pt)):
            extraSpring_vert.append([self.target_pt[i],self.pts[i]])
        self.extraSpring = extraSpring_vert
        ...

    def get_pts(self):
        return self.pts
    
    def get_target_pt(self):
        return self.target_pt
    
    def get_target_pos(self):
        return self.target_pos
    
    def get_mass(self): # this is on particle
        return self.mass
    
    def get_stiffness(self):
        return self.stiffness
    
    def get_restlength(self):
        return self.restlength

    def parse_primattributes(self):
        if 'primitiveattributes' not in self.attributes:
            return
        
        self.primitiveattributes = self.attributes['primitiveattributes']
        class AttributeValue:
            None
        class PrimAttr:
            None
        # parse point attributes
        allPrimAttr = []
        for i in range(len(self.primitiveattributes)):
            attrRaw0 = self.primitiveattributes[i][0] #metadata
            a = PrimAttr()
            for name,item in zip(attrRaw0[0::2],attrRaw0[1::2]):
                a.__setattr__(name,item)

            attrRaw1 = self.primitiveattributes[i][1] #data
            for name,item in zip(attrRaw1[0::2],attrRaw1[1::2]):
                a.__setattr__(name,item)

            if a.name == "extraSpring":
                extraSpring_dict = a.dicts
                self.parse_extraSpring_from_dict(extraSpring_dict)
            if a.name == "target_pt":
                self.target_pt = a.values[5][0]
            if a.name == "pts":
                self.pts = a.values[5][0]
            if a.name == "stiffness":
                self.stiffness = a.values[5][0]
            if a.name == "restlength":
                self.restlength = a.values[5][0]
            if a.name == "target_pos":
                self.target_pos = a.values[5]
            allPrimAttr.append(a)

        

    
class Polygon(object):
    def __init__(self,indices,closed=False):
        self.indicies = indices
        self.closed   = closed

def read_geo(input):
    geo = Geo(input)
    return geo

def test_geo_vtk():
    dir = str(Path(__file__).parent) + "/"
    geo = read_geo(dir+"sample_in.geo")
    geo.read_vtk(dir+"sample.vtk")
    geo.write("sample_out.geo")


def test_animation():
    dir = str(Path(__file__).parent.parent.parent) + "/" + "data/model/pintoanimation/"
    geo = read_geo(dir+"physdata_78.geo")
    pin = geo.get_gluetoaniamtion()
    vert = geo.get_vert()
    pos = geo.get_pos()
    pos[0] = [0,0,0]
    pos[1] = [0,0,0]
    pos[2] = [0,0,0]
    geo.set_positions(pos)
    geo.write(dir+"physdata_78_out.geo")
    

if __name__ == '__main__':
    test_animation()