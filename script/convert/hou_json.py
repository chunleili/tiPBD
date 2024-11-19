""" A easy to parse json format for IO in houdini"""
import meshio
from pathlib import Path
import json
from dataclasses import dataclass, field

@dataclass
class Point:
    positions: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


@dataclass
class Geometry():
    def __init__(self):
        self.points = []
        self.prims = []
        self.indices = []
        self.POLYGON_TYPE = 3 # default triangle
    
    def pointcount(self):
        return len(self.points)
    def primitivecount(self):
        return len(self.prims)

    # TODO: Now we only have triangle and one attribute
    def read_vtk(self,input:str):
        self.input = input
        mesh = meshio.read(input, file_format="vtk")
        self.points = Point(mesh.points.tolist())
        tri = mesh.cells_dict['triangle']
        self.POLYGON_TYPE = tri.shape[1]
        self.indices = tri.flatten().tolist()
        for k,v in mesh.cell_data.items():
            for i in range(len(v)):
                self.prims.append({k:v[i].tolist()})

    def __str__(self):
        return f"Points: {self.points}\nPrimitives: {self.prims}\nIndices: {self.indices}"

    def to_json(self):
        j = dict()
        j["points"] = {}
        j["points"]["P"] = self.points.positions
        j["points"]["attributes"] = self.points.attributes
        j["primitives"] = self.prims
        j["indices"] = self.indices
        j["POLYGON_TYPE"] = self.POLYGON_TYPE
        s = json.dumps(j, indent=4)
        return s

    def write_json(self, output:str=None):
        if output:
            self.output = output
        else:
            self.output = str(Path(self.input).parent) + "/" + str(Path(self.input).stem) + ".json"

        j = self.to_json()
        with open(self.output, "w") as f:
            f.write(j)
        print("Finish writing json file: ", self.output)


def test():
    import os
    dir = os.path.dirname(os.path.realpath(__file__))
    geo = Geometry()
    geo.read_vtk(dir+"/sample.vtk")
    print(geo.points)
    print(geo.indices)
    print(geo.prims)
    geo.write_json()

if __name__ == "__main__":
    test()