import numpy as np
import taichi as ti
from enum import Enum
class ConstraintType(Enum):
    STRETCH = 0
    ATTACHMENT = 1
    ΒENDING = 2
    def __str__(self):
        return self.name

class Constraint:
    def __init__(self, stiffness):
        self.stiffness = stiffness

class SpringConstraint(Constraint):
    def __init__(self, stiffness:float, p1:int, p2:int, rest_len:float, type=ConstraintType.STRETCH):
        super().__init__(stiffness)
        self.p1 = p1
        self.p2 = p2
        self.rest_len = rest_len
        self.type = type

    def __str__(self):
        return f"SpringConstraint: {self.p1} - {self.p2} rest_len: {self.rest_len:.10g} stiffness: {self.stiffness} type: {self.type.name.lower()}"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SpringConstraint):
            return False
        if self.p1 != value.p1:
            return False
        if self.p2 != value.p2:
            return False
        if self.rest_len != value.rest_len:
            return False
        if self.stiffness != value.stiffness:
            return False
        if self.type != value.type:
            return False
        return True


class AttachmentConstraint(Constraint):
    def __init__(self, stiffness:float, p0:int, fixed_point:np.ndarray):
        super().__init__(stiffness)
        self.p0 = p0
        assert fixed_point.shape == (3,)
        self.fixed_point = fixed_point
        self.type = ConstraintType.ATTACHMENT

    def __str__(self):
        return f"AttachmentConstraint: {self.p0} fixed_point: {self.fixed_point} stiffness: {self.stiffness} type: {self.type.name.lower()}"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AttachmentConstraint):
            return False
        if self.p0 != value.p0:
            return False
        if self.type != value.type:
            return False
        if not np.allclose(self.fixed_point, value.fixed_point):
            return False
        if self.stiffness != value.stiffness:
            return False
        return True

class Mesh:
    def __init__(self, dim, pos, edge):
        self.dim = dim
        self.current_positions = pos
        self.edge_list = edge
        

class SetupConstraints:
    def __init__(self, pos, edge):
        self.mesh = Mesh((21, 21), pos, edge)
        self.stiffness_stretch = 80.0
        self.stiffness_bending = 20.0
        self.stiffness_attachment = 120.0
        self.constraints = []

    def add_attachment_constraint(self, vertex_index):
        ac = AttachmentConstraint(self.stiffness_attachment, vertex_index, self.mesh.current_positions[vertex_index])
        self.constraints.append(ac)

                
    def setup_constraints(self):
        # generate attachment constraints.
        self.add_attachment_constraint(0)
        self.add_attachment_constraint(self.mesh.dim[1] * (self.mesh.dim[0] - 1))

        # generate stretch constraints. assign a stretch constraint for each edge.
        for e in self.mesh.edge_list:
            p1 = self.mesh.current_positions[e[0]]
            p2 = self.mesh.current_positions[e[1]]
            c = SpringConstraint(self.stiffness_stretch, e[0], e[1], np.linalg.norm(p1 - p2))
            self.constraints.append(c)

        # generate bending constraints. naive
        for i in range(self.mesh.dim[0]):
            for k in range(self.mesh.dim[1]):
                index_self = self.mesh.dim[1] * i + k
                p1 = self.mesh.current_positions[index_self]
                if i + 2 < self.mesh.dim[0]:
                    index_row_1 = self.mesh.dim[1] * (i + 2) + k
                    p2 = self.mesh.current_positions[index_row_1]
                    c = SpringConstraint(self.stiffness_bending, index_self, index_row_1, np.linalg.norm(p1 - p2), type=ConstraintType.ΒENDING)
                    self.constraints.append(c)
                if k + 2 < self.mesh.dim[1]:
                    index_column_1 = self.mesh.dim[1] * i + k + 2
                    p2 = self.mesh.current_positions[index_column_1]
                    c = SpringConstraint(self.stiffness_bending, index_self, index_column_1, np.linalg.norm(p1 - p2), type=ConstraintType.ΒENDING)
                    self.constraints.append(c)
        return self.constraints
    
    
    def read_constraints(self,file):
        constraints = []
        with open(file, "rt") as f:
            lines = f.readlines()
            for line in lines:
                if "SpringConstraint" in line:
                    parts = line.split()
                    p1 = int(parts[1])
                    p2 = int(parts[3])
                    rest_len = float(parts[5])
                    stiffness = float(parts[7])
                    if parts[9] == "bending":
                        type = ConstraintType.ΒENDING
                    else:
                        type = ConstraintType.STRETCH
                    c = SpringConstraint(stiffness, p1, p2, rest_len, type)
                    constraints.append(c)
                elif "AttachmentConstraint" in line:
                    line = line.replace("(", "")
                    line = line.replace(")", "")
                    line = line.replace(",", " ")
                    parts = line.split()
                    for i,p in enumerate(parts):
                        if p == "vertex":
                            p0 = int(parts[i+1])
                        elif p == "fixed_point:":
                            fixed_point = np.array([float(parts[i+1]), float(parts[i+2]), float(parts[i+3])])
                        elif p == "stiffness:":
                            stiffness = float(parts[i+1])
                        
                    c = AttachmentConstraint(stiffness, p0, fixed_point)
                    constraints.append(c)
        return constraints    
    