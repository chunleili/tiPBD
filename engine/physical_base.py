"""Base class for all physical solver"""

import taichi as ti

from engine.util import ResidualDataAllFrame, ResidualDataOneFrame, ResidualDataOneIter
from engine.physical_data import PhysicalData
@ti.data_oriented
class PhysicalBase:
    def __init__(self) -> None:
        self.frame = 0
        self.ite = 0
        self.n_outer_all = [] 
        self.all_stalled = [] 
        self.r_frame = ResidualDataOneFrame([])
        self.r_all = ResidualDataAllFrame([],[])


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
        raise NotImplementedError
    
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

@ti.kernel
def compute_potential_energy_kernel(
    constraints: ti.template(),
    alpha_tilde: ti.template(),
    delta_t: ti.f32,
)->ti.f32:
    potential_energy = 0.0
    for i in range(constraints.shape[0]):
        inv_alpha =  1.0 / (alpha_tilde[i]*delta_t**2)
        potential_energy += 0.5 * inv_alpha * constraints[i]**2
    return potential_energy

@ti.kernel
def compute_inertial_energy_kernel(
    pos: ti.template(),
    predict_pos: ti.template(),
    inv_mass: ti.template(),
    delta_t: ti.f32,
)->ti.f32:
    inertial_energy = 0.0
    inv_h2 = 1.0 / delta_t**2
    for i in range(pos.shape[0]):
        if inv_mass[i] == 0.0:
            continue
        inertial_energy += 0.5 / inv_mass[i] * (pos[i] - predict_pos[i]).norm_sqr() * inv_h2
    return inertial_energy