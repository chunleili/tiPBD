import numpy as np

class PhysicalData:
    def __init__(self, pos=None, vel=None, stiffness=None, rest_len=None, vert=None, mass=None, delta_t=None, force=None, pin=None, pinpos=None, restmatrix=None, *pargs, **kwargs):
        """
        Parameters
        ----------
        pos : np.ndarray(dtype=np.float32, shape=(NV, 3))
            position of the vertices
        vel : np.ndarray(dtype=np.float32, shape=(NV, 3))
            velocity of the vertices
        pos_old : np.ndarray(dtype=np.float32, shape=(NV, 3))
            last substep position of the vertices
        stiffness : np.ndarray(dtype=np.float32, shape=(NCONS,))
            stiffness of the constraints
        rest_len : np.ndarray(dtype=np.float32, shape=(NCONS,))
            rest length of the constraints
        vert : np.ndarray(dtype=np.int32, shape=(NCONS, NVERTS_ONE_CONS))
            vertex index for each constraint
        mass : np.ndarray(dtype=np.float32, shape=(NV,))
            mass of the vertices
        delta_t : float
            time step size
        force : np.ndarray(dtype=np.float32, shape=(NV, 3))
            external force applied to the vertices
        pin : np.ndarray(dtype=np.int32, shape=(NV,))
            0: free, 1: pinned
        pinpos: np.ndarray(dtype=np.float32, shape=(NV, 3))
            key frame positions, only pinned vertices will be used
        restmatrix: np.ndarray(dtype=np.float32, shape=(NCONS, 3, 3))
            rest matrix of the vertices
        """
        self.pos = pos
        self.vel = vel
        self.stiffness = stiffness
        self.rest_len = rest_len
        self.vert = vert # vertex index for each constraint
        self.mass = mass
        self.delta_t = delta_t
        self.force = force
        self.NV = self.mass.shape[0] if self.mass is not None else None
        self.NCONS = self.stiffness.shape[0] if self.stiffness is not None else None
        self.NVERTS_ONE_CONS = self.vert.shape[1] if self.vert is not None else None

        self.pin = pin


    def read_json(self, json_path):
        """
        Parameters
        ----------
        json_path : str
            path to the json file

            Contains the following keys:
            - pos
            - vel
            - stiffness
            - rest_len
            - vert
            - mass
            - delta_t
            - force
            - pin
            - pinpos
            - restmatrix
            - type (optional)
            - predict_pos (optional)
        """
        import json
        with open(json_path, "rt") as f:
            data = json.load(f)
        self.pos = np.array(data["pos"], dtype=np.float32)
        self.vel = np.array(data["vel"], dtype=np.float32)
        self.stiffness = np.array(data["stiffness"], dtype=np.float32)
        self.rest_len = np.array(data["rest_len"], dtype=np.float32)
        self.vert = np.array(data["vert"], dtype=np.int32)
        self.mass = np.array(data["mass"], dtype=np.float32)
        self.delta_t = data["delta_t"]
        self.force = np.array(data["force"], dtype=np.float32)
        self.pin = data["pin"]
        self.NV = self.mass.shape[0]
        self.NCONS = self.stiffness.shape[0]
        self.NVERTS_ONE_CONS = self.vert.shape[1]

        # optional keys
        if "predict_pos" in data:
            self.predict_pos = np.array(data["predict_pos"], dtype=np.float32)

    
    def write_json(self, json_path):
        """
        Parameters
        ----------
        json_path : str
            path to the json file
        """
        import json
        # mandatory keys
        data = {
            "pos": self.pos.tolist(),
            "vel": self.vel.tolist(),
            "stiffness": self.stiffness.tolist(),
            "rest_len": self.rest_len.tolist(),
            "vert": self.vert.tolist(),
            "mass": self.mass.tolist(),
            "delta_t": self.delta_t,
            "force": self.force.tolist(),
            "pin": self.pin.tolist()
        }
        # optional keys
        if hasattr(self, "predict_pos"):
            data["predict_pos"] = self.predict_pos.tolist()

        with open(json_path, "wt") as f:
            json.dump(data, f, indent=4)

    def to_taichi_fields(self):
        import taichi as ti
        NVERTS_ONE_CONS = self.vert.shape[1]

        # back up the numpy arrays
        self.stiffness_np = self.stiffness.copy()
        self.rest_len_np = self.rest_len.copy()
        self.vert_np = self.vert.copy()
        self.mass_np = self.mass.copy()
        self.force_np = self.force.copy()

        # allocate fields
        self.pos = ti.Vector.field(3, dtype=float, shape=self.NV)
        self.stiffness = ti.field(dtype=float, shape=self.NCONS)
        self.rest_len = ti.field(dtype=float, shape=self.NCONS)
        self.vert = ti.Vector.field(NVERTS_ONE_CONS, dtype=int, shape=self.NCONS)
        self.mass = ti.field(dtype=float, shape=self.NV)
        self.force = ti.Vector.field(3, dtype=float, shape=self.NV)

        # copy the numpy arrays to the fields
        self.pos.from_numpy(self.pos_np)
        self.stiffness.from_numpy(self.stiffness_np)
        self.rest_len.from_numpy(self.rest_len_np)
        self.vert.from_numpy(self.vert_np)
        self.mass.from_numpy(self.mass_np)
        self.force.from_numpy(self.force_np)

        