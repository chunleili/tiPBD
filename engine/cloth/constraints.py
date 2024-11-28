import numpy as np
import taichi as ti
import scipy
from enum import Enum
import taichi.math as tm
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
    def __init__(self, pos:np.ndarray, edge:np.ndarray, N=20, stiffness_stretch=120.0, stiffness_bending=20.0, stiffness_attachment=120.0, fixed_points_num=[0,410], use_bending=True):
        # self.mesh = Mesh((21, 21), pos, edge)
        self.mesh = Mesh((N+1, N+1), pos, edge)
        self.stiffness_stretch = stiffness_stretch
        self.stiffness_bending = stiffness_bending
        self.stiffness_attachment = self.stiffness_stretch
        self.use_bending = use_bending
        self.fixed_points_num = [0, self.mesh.dim[1] * (self.mesh.dim[0] - 1)]
        self.constraints = []
        # self.setup_constraints()
        self.read_constraints("constraints.txt")
        self.adapter = constraintsAdapter(self.constraints) #FIXME: no attachment now
        

    def add_attachment_constraint(self, vertex_index):
        ac = AttachmentConstraint(self.stiffness_attachment, vertex_index, self.mesh.current_positions[vertex_index])
        self.constraints.append(ac)

                
    def setup_constraints(self):
        # generate attachment constraints.
        # for idx in self.fixed_points_num:
        #     self.add_attachment_constraint(idx)

        # self.add_attachment_constraint(0)
        # self.add_attachment_constraint(self.mesh.dim[1] * (self.mesh.dim[0] - 1))

        # generate stretch constraints. assign a stretch constraint for each edge.
        for e in self.mesh.edge_list:
            p1 = self.mesh.current_positions[e[0]]
            p2 = self.mesh.current_positions[e[1]]
            c = SpringConstraint(self.stiffness_stretch, e[0], e[1], np.linalg.norm(p1 - p2))
            self.constraints.append(c)

        if not self.use_bending:
            return self.constraints

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
        self.constraints = constraints
        return constraints    
    


@ti.data_oriented
class constraintsAdapter:
    def __init__(self, constraintsNew):
        NVERTS_ONE_CONS = 2
        self.NVERTS_ONE_CONS = NVERTS_ONE_CONS

        self.constraintsNew = constraintsNew
        NCONS = len(constraintsNew)
        self.NCONS = NCONS
        self.val_np = np.zeros(dtype=np.float32, shape=NCONS)
        self.rest_len_np = np.zeros(dtype=np.float32, shape=NCONS)
        self.vert_np = np.zeros(dtype=np.int32, shape=(NCONS, NVERTS_ONE_CONS)) 
        self.stiffness_np = np.zeros(dtype=np.float32, shape=NCONS)
        self.cType = np.zeros(dtype=np.int32, shape=NCONS) # 0: stretch, 1: attachment, 2: bending
        self.p0 = np.zeros(dtype=np.int32, shape=NCONS)
        self.fixed_point = np.zeros(dtype=np.float32, shape=(NCONS, 3))
        self.pinlist = []
        self.pinposlist = []
        self.list_to_ndarray()


        self.val = ti.field(dtype=float, shape=NCONS)
        self.rest_len = ti.field(dtype=float, shape=NCONS)
        self.vert = ti.Vector.field(NVERTS_ONE_CONS, dtype=int, shape=NCONS)
        self.stiffness = ti.field(dtype=float, shape=NCONS)
        self.ndarray_to_fields()
    

    def list_to_ndarray(self):
        for i in range(self.NCONS):
            if isinstance(self.constraintsNew[i], AttachmentConstraint):
                self.stiffness_np[i] = self.constraintsNew[i].stiffness
                self.pinposlist.append(self.constraintsNew[i].fixed_point)
                self.pinlist.append(self.constraintsNew[i].p0)
                self.cType[i] = 1
                self.p0[i] = self.constraintsNew[i].p0
                self.fixed_point[i] = self.constraintsNew[i].fixed_point
                continue
            self.rest_len_np[i] = self.constraintsNew[i].rest_len
            self.vert_np[i, 0] = self.constraintsNew[i].p1
            self.vert_np[i, 1] = self.constraintsNew[i].p2
            self.stiffness_np[i] = self.constraintsNew[i].stiffness

    def ndarray_to_fields(self):
        self.val.from_numpy(self.val_np)
        self.rest_len.from_numpy(self.rest_len_np)
        self.vert.from_numpy(self.vert_np)
        self.stiffness.from_numpy(self.stiffness_np)



@ti.dataclass
class DistanceConstraintAOS:
    idx: ti.i32
    c: ti.f32
    g1=ti.math.vec3([0,0,0])
    g2=ti.math.vec3([0,0,0])
    lam:ti.f32
    dlam:ti.f32
    p1:ti.i32
    p2:ti.i32
    typeid: ti.i32
    stiffness: ti.f32
    alpha_tilde:ti.f32
    rest_len: ti.f32

    @ti.func
    def calc_rest_len(self,pos:ti.template()):
        self.rest_len = (pos[self.p1] - pos[self.p2]).norm()




class ClothConstraints():
    def __init__(self, args, vert, pos, tri=None, ):
        if args.use_bending:
            from engine.cloth.bending import init_bending, add_distance_constraints_from_tri_pairs
            self.tri_pairs, self.bending_length = init_bending(tri, pos)
            self.vert = add_distance_constraints_from_tri_pairs(vert, self.tri_pairs)

        NCONS = vert.shape[0]

        self.rest_len = self.init_rest_len(vert, pos)
        self.set_alpha(args.compliance, NCONS, args.delta_t)
        self.disConsAos =  self.to_AOS_constraints(pos,vert,self.alpha_tilde)

        self.lagrangian  = ti.field(dtype=float, shape=(NCONS))  
        self.constraints = ti.field(dtype=float, shape=(NCONS))  
        self.dLambda     = ti.field(dtype=float, shape=(NCONS))
        self.gradC       = ti.Vector.field(3, dtype = ti.float32, shape=(NCONS,2)) 
        self.dual_residual= ti.field(shape=(NCONS),    dtype = ti.float32) # -C-alpha_tilde * self.lagrangian
 
        self.cType = np.zeros(NCONS, dtype=np.int32) # 0: stretch, 1: attachment, 2: bending
        self.p0 = np.zeros(NCONS, dtype=np.int32)
        self.fixed_point = np.zeros((NCONS, 3), dtype=np.float32)

    
    def set_alpha(self, compliance, NCONS, delta_t):
        alpha_tilde_constant = compliance * (1.0 / delta_t / delta_t)  
        # timestep related compliance, see XPBD paper 
        # #see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
        alpha_bending_constant = 1.0 * (1.0 / delta_t / delta_t) #TODO: need to be tuned
        stiffness = ti.field(dtype=float, shape=(NCONS))
        stiffness.fill(1.0/compliance)
        alpha_tilde_np = np.array([alpha_tilde_constant] * NCONS)

        # self.ALPHA_TILDE = scipy.sparse.diags(alpha_tilde_np)
        self.stiffness_matrix = scipy.sparse.dia_matrix((stiffness.to_numpy(), [0]), shape=(NCONS, NCONS), dtype=np.float32)
        self.alpha_tilde = ti.field(dtype=ti.f32, shape=(NCONS))
        self.alpha_tilde.from_numpy(alpha_tilde_np)


    def init_rest_len(self, vert, pos):
        @ti.kernel
        def kernel(
            vert:ti.template(),
            pos:ti.template(),
            rest_len:ti.template(),
        ):
            for i in range(vert.shape[0]):
                idx1, idx2 = vert[i]
                p1, p2 = pos[idx1], pos[idx2]
                rest_len[i] = (p1 - p2).norm()
        rest_len = ti.field(dtype=float, shape=(vert.shape[0]))
        kernel(vert, pos, rest_len)
        return rest_len

    def to_AOS_constraints(self, pos, vert, alpha_tilde):
        @ti.kernel
        def kernel(
            pos:ti.template(),
            vert:ti.template(),
            alpha_tilde:ti.template(),
            aos: ti.template(),
        ):
            for i in range(vert.shape[0]):
                aos[i].p1, aos[i].p2 = vert[i]
                aos[i].calc_rest_len(pos)
                aos[i].alpha_tilde = alpha_tilde[i]
        
        disConsAos = DistanceConstraintAOS.field(shape=vert.shape[0])
        kernel(pos,vert,alpha_tilde,disConsAos)
        return disConsAos