from engine.metadata import meta
import numpy as np
def log_energy(mesh, write_energy_to_file=True):
    if write_energy_to_file and meta.frame%100==0:
        print(f"frame: {meta.frame} potential: {mesh.potential_energy[None]:.3e} inertial: {mesh.inertial_energy[None]:.3e} total: {mesh.total_energy[None]:.3e}")
        with open(meta.result_path+"/totalEnergy.txt", "ab") as f:
            np.savetxt(f, np.array([mesh.total_energy[None]]), fmt="%.4e", delimiter="\t")
        with open(meta.result_path+"/potentialEnergy.txt", "ab") as f:
            np.savetxt(f, np.array([mesh.potential_energy[None]]), fmt="%.4e", delimiter="\t")
        with open(meta.result_path+"/inertialEnergy.txt", "ab") as f:
            np.savetxt(f, np.array([mesh.inertial_energy[None]]), fmt="%.4e", delimiter="\t")