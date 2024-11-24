"""Base class for all physical solver"""

import taichi as ti
import datetime, logging
import numpy as np

from engine.util import ResidualDataAllFrame, ResidualDataOneFrame, ResidualDataOneIter
from engine.physical_data import PhysicalData
from engine.ti_kernels import *
@ti.data_oriented
class PhysicalBase:
    def __init__(self) -> None:
        self.frame = 0
        self.ite = 0
        self.n_outer_all = [] 
        self.all_stalled = [] 
        self.r_frame = ResidualDataOneFrame([])
        self.r_all = ResidualDataAllFrame([],[])
        self.start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logging.info(f"start date:{self.start_date}")

    def do_pre_iter0(self):
        self.update_constraints() # for calculation of r0
        self.r_iter.calc_r0()
        
    def do_post_iter_xpbd(self):
        # self.update_constraints()
        self.r_iter.calc_r(self.frame,self.ite, self.r_iter.tic_iter)
        self.r_all.t_export += self.r_iter.t_export
        self.r_iter.t_export = 0.0
    
    def update_pos(self):
        update_pos_kernel(self.inv_mass, self.dpos, self.pos, self.omega)

    def collision_response(self):
        if self.args.use_ground_collision:
            ground_collision_kernel(self.pos, self.old_pos, self.ground_pos, self.inv_mass)

    def semi_euler(self):
        semi_euler_kernel(self.delta_t, self.pos, self.predict_pos, self.old_pos, self.vel, self.damping_coeff, self.gravity, self.inv_mass, self.force)

    def update_vel(self):
        update_vel_kernel(self.delta_t, self.pos, self.old_pos, self.vel, self.inv_mass)


    def to_physdata(self, physdata):
        physdata.pos = self.pos.to_numpy()
        physdata.stiffness = self.stiffness.to_numpy()
        physdata.rest_len = self.rest_len.to_numpy()
        physdata.vert = self.vert.to_numpy()
        physdata.mass = self.mass.to_numpy()
        physdata.delta_t = self.delta_t
        physdata.force = self.force.to_numpy()
        if not hasattr(self.args, "physdata_json_file"):
            self.args.physdata_json_file = "physdata.json"
        physdata.write_json(self.args.physdata_json_file)


    def calc_dual(self):
        dual = calc_dual_kernel(self.dual_residual, self.lagrangian, self.constraints, self.dual_residual)
        return dual
    
    def calc_primal(self):
        raise NotImplementedError
    
    def calc_strain(self):
        raise NotImplementedError

    def calc_energy(self):
        self.update_constraints()
        self.potential_energy = self.compute_potential_energy()
        self.inertial_energy = self.compute_inertial_energy()
        self.energy = self.potential_energy + self.inertial_energy
        return self.energy

    def compute_potential_energy(self)->float:
        res = compute_potential_energy_kernel(self.constraints, self.alpha_tilde, self.delta_t)
        return res

    def compute_inertial_energy(self)->float:
        res = compute_inertial_energy_kernel(self.pos, self.predict_pos, self.inv_mass, self.delta_t)
        return res
    
    def should_setup(self):
        if self.ite != 0:
            return False
        if self.frame==self.initial_frame:
            return True
        if self.frame%self.args.setup_interval==0:
            return True
        if self.args.restart and self.frame==self.initial_frame:
            return True
        return False
