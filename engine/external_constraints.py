import os
import sys
import numpy as np
import taichi as ti


prj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_path)
from engine.util import timeit, python_list_to_ti_field
from script.convert.geo import Geo


class ExternalConstraints:
    """
    Class to handle external constraints for muscle solvers.
    Usage:  
            Initialize:
            from engine.external_constraints import ExternalConstraints
            self.muscle = ExternalConstraints(args, self)

            Before the simulation loop:
            if args.use_external_constraints:
                self.pos = self.muscle.handle_external_constraints(self.frame, self.pos)
    """

    def __init__(self, args):
        """
        Initialize the ExternalConstraints class.

        Parameters:
        - args: Command line arguments containing simulation parameters.
        """
        self.args = args
        args.use_external_constraints = True
        args.use_houdini_data= True
        self.args.use_extra_spring = False #TODO
        self.args.use_muscle2muscle = False #TODO
        self.args.use_pintoanimation = False #TODO
        if not hasattr(self.args, "start_frame"):
            self.args.start_frame = 1

        self.initialize() 

    def initialize(self):
        self.read_geo_rest() 
        if self.args.use_extra_spring:
            self.read_extra_spring_rest()
        if self.args.use_pintotarget:
            self.read_pintotarget_rest()
        if self.args.use_muscle2muscle:
            self.read_muscle2muscle_rest()

    def read_extra_spring_rest(self):
        dir = prj_path + "/" + self.args.geo_dir + "/"
        consgeo = Geo(dir + f"cons_{self.args.start_frame}.geo")
        self.consgeo_rest = consgeo

        # read connectivity
        # first column is target(driving point), second column is source(driven)
        pts1 = np.array(consgeo.get_pts())
        pts = ti.field(int, pts1.shape[0])
        pts.from_numpy(pts1)

        # read sim pos(to be driven)
        pos1 = np.array(self.geo_rest.get_pos(), dtype=np.float32)
        pos_ = ti.Vector.field(3, ti.f32, pos1.shape[0])
        pos_.from_numpy(pos1)

        # read target pos(driving)
        tp = consgeo.get_target_pos()
        self.target_pos = python_list_to_ti_field(tp)

        from engine.constraints.distance_constraints import DistanceConstraintsAttach

        self.extra_springs = DistanceConstraintsAttach(pts, pos_, self.target_pos)

        # optional data(inv_mass, stiffness, restlength)
        # self.extra_springs.set_alpha(consgeo.get_stiffness())
        # self.extra_springs.set_rest_len(consgeo.get_restlength())

    def read_muscle2muscle_rest(self, pos):
        """read muscle2muscle topology from geo file
        It start from pt_index 1
        """
        dir = prj_path + "/" + self.args.geo_dir + "/"
        m2mgeo = Geo(dir + f"m2m.geo")
        self.m2mgeo = m2mgeo

        # source pt(interior pt of muscle A)
        src = np.array(m2mgeo.get_pts())

        # target pts (surface pts of muscle B, could be multiple)
        tps = m2mgeo.get_target_pts()

        # pairs of (p1, p2)
        pairs = []
        for i, p1 in enumerate(src):
            p2s = tps[i]
            for k, p2 in enumerate(p2s):
                pairs.append((p1, p2))

        pairs_np = np.array(pairs)
        p1 = python_list_to_ti_field(pairs_np[:, 0].tolist())
        p2 = python_list_to_ti_field(pairs_np[:, 1].tolist())
        from engine.constraints.distance_constraints import DistanceConstraints

        self.m2mCons = DistanceConstraints(p1, p2, pos)

    def read_pintotarget_rest(self):
        dir = prj_path + "/" + self.args.geo_dir + "/"
        consgeo = Geo(dir + f"cons_{self.args.start_frame}.geo")
        self.consgeo_rest = consgeo

        # read connectivity
        # target_pos is driving point, pts is source points(to be driven)
        pts1 = np.array(consgeo.get_pts())
        pts = ti.field(int, pts1.shape[0])
        pts.from_numpy(pts1)

        # read sim pos(to be driven)
        pos1 = np.array(self.geo_rest.get_pos(), dtype=np.float32)
        pos = ti.Vector.field(3, ti.f32, pos1.shape[0])
        pos.from_numpy(pos1)

        # read target pos(driving)
        target_pos = np.array(consgeo.get_target_pos(), dtype=np.float32)
        self.target_pos = ti.Vector.field(3, ti.f32, target_pos.shape[0])
        self.target_pos.from_numpy(target_pos)

        from engine.constraints.distance_constraints import PinToTarget

        self.pintotarget = PinToTarget(pts, pos, self.target_pos)

    def read_target_pos(self, frame):
        dir = prj_path + "/" + self.args.geo_dir + "/"
        geo = Geo(dir + f"cons_{frame}.geo")
        tp = np.array(geo.get_target_pos(), dtype=np.float32)
        self.target_pos.from_numpy(np.array(tp, dtype=np.float32))
        ...

    def read_geo_pinpos(self, frame, pos):
        dir = prj_path + "/" + self.args.geo_dir + "/"
        geo = Geo(dir + f"physdata_{frame}.geo")
        pinpos = np.array(geo.get_pos())
        assert pinpos.shape[0] == pos.shape[0]
        # set_pinpos_kernel(self.pin, self.pos, pinpos)

        self.pinlist = np.where(self.pin)[0]
        self.inv_mass_np = self.inv_mass.to_numpy()
        self.inv_mass_np[self.pinlist] = 0.0
        self.inv_mass.from_numpy(self.inv_mass_np)

        pos_ = pos.to_numpy()
        pos_[self.pin] = pinpos[self.pin]
        pos.from_numpy(pos_)


    def read_geo_rest(self, filename="restpos.geo"):
        dir = prj_path + "/" + self.args.geo_dir + "/"
        filename = dir + filename
        if os.path.exists(filename):
            geo = Geo(filename)
        else:
            raise FileNotFoundError(f"restpos.geo not found")

        self.pin = np.array(geo.get_gluetoaniamtion(), dtype=np.bool_)
        self.vert = np.array(geo.get_vert(), dtype=np.int32)
        self.pos_rest = np.array(geo.get_pos(), dtype=np.float32)

        self.NV = self.pos_rest.shape[0]
        self.NT = self.vert.shape[0]

        self.geo_dir = dir
        self.geo = geo
        self.geo_rest = geo


    def fetch_fields(self, ist):
        ist.NV = self.NV
        ist.NT = self.NT
        # ist.allocate_fields(self.NV, self.NT)
        # ist.inv_mass.from_numpy(im)
        ist.pos.from_numpy(self.pos)
        ist.tet_indices.from_numpy(self.pos)
        ist.geo = self.pos

        # # read mass from geo
        # im = np.array(geo.get_mass(), dtype=np.float32)
        # im = 1.0 / im[np.isnan(im) == False]
        # # set pinned point inv_mass to 0
        # im[pin] = 0.0

        # # TODO: TO BE REMOVED.  Transfering the data reference between self and ist
        # ist.NV = self.NV
        # ist.NT = self.NT
        # ist.allocate_fields(self.NV, self.NT)
        # # ist.inv_mass.from_numpy(im)
        # ist.pos.from_numpy(pos_)
        # ist.tet_indices.from_numpy(vert)
        # ist.geo = geo
        # return self.NV, self.NT,  vert, pos_, im, geo, pin


    def write_geo(self, output=None):
        self.geo.set_positions(self.pos.to_numpy())
        if output is None:
            output = self.geo_dir + f"physdata_{self.frame}_out.geo"
        self.geo.write(output)

    # @timeit
    def read_external_pos(self,frame,pos):
        if self.args.use_pintoanimation:
            self.read_geo_pinpos(frame,pos)
        if self.args.use_extra_spring or self.args.use_pintotarget:
            self.read_target_pos(frame)

    # # @timeit
    def do_external_constraints(self, pos):
        if self.args.use_extra_spring:
            self.extra_springs.aos.lam.fill(0.0)
            self.extra_springs.solve_one_iter(pos, self.target_pos, self.args.delta_t)
        if self.args.use_pintotarget:
            self.pintotarget.solve(pos, self.target_pos, 1)
        if self.args.use_muscle2muscle:
            self.m2mCons.aos.lam.fill(0.0)
            self.m2mCons.solve_one_iter(pos, self.args.delta_t)

    def handle_external_constraints(self, frame, pos):
        """
        Call this function every substep to update the external constraints.
        It will read the external positions and apply the constraints.

        Parameters:
        - pos: taichi field containing the current positions of the points.
        """
        self.read_external_pos(frame,pos)
        self.do_external_constraints(pos)
        return pos
