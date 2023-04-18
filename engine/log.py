from engine.metadata import meta
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
def log_energy(model, write_energy_to_file=False, interval=100):
    if meta.frame%interval==0:
        # print(f"frame: {meta.frame} potential: {model.potential_energy[None]:.3e} inertial: {model.inertial_energy[None]:.3e} total: {model.total_energy[None]:.3e}")
        logging.info(f"frame: {meta.frame} potential: {model.potential_energy[None]:.3e} inertial: {model.inertial_energy[None]:.3e} total: {model.total_energy[None]:.3e}")

        if write_energy_to_file:
            with open(meta.result_path+"/totalEnergy.txt", "ab") as f:
                np.savetxt(f, np.array([model.total_energy[None]]), fmt="%.4e", delimiter="\t")
            with open(meta.result_path+"/potentialEnergy.txt", "ab") as f:
                np.savetxt(f, np.array([model.potential_energy[None]]), fmt="%.4e", delimiter="\t")
            with open(meta.result_path+"/inertialEnergy.txt", "ab") as f:
                np.savetxt(f, np.array([model.inertial_energy[None]]), fmt="%.4e", delimiter="\t")