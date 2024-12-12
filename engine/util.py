import taichi as ti
import numpy as np    
import logging
import json
from time import perf_counter
from dataclasses import dataclass, field
from enum import Enum
import scipy


class ResidualType(Enum):
    dual = 1
    primal = 2
    energy = 3
    strain = 4
    Newton = 5

@dataclass
class ResidualDataOneIter:
    name = "residualOneIter"
    ninner: int=0
    t_iter: float=0
    dual: float=0
    primal: float=0
    Newton: float=0
    dual0: float=0
    primal0: float=0
    Newton0: float=0
    r_Axb: list[float] = field(default_factory=list)
    frame: int=0
    ite: int=0
    # r: float=0
    # r0: float=0
    t_export: float=0
    energy: float=0
    energy0: float=0
    max_strain: float=0
    max_strain0: float=0

    def __init__(
        self,
        calc_dual=None,
        calc_primal=None,
        calc_energy=None,
        calc_strain=None,
        tol=1e-6,
        rtol=1e-2,
        converge_condition="dual",
        args = None
    ):
        self.tol = tol
        self.rtol = rtol
        self.calc_dual = calc_dual
        self.calc_primal = calc_primal
        self.calc_energy = calc_energy
        self.calc_strain = calc_strain
        if args is not None:
            self.use_calc_dual   = args.calc_dual
            self.use_calc_primal = args.calc_primal
            self.use_calc_energy = args.calc_energy
            self.use_calc_strain = args.calc_strain
        self.choose_mode(converge_condition)

    def check(self):
        '''Check Convergence'''
        self.set_r() # set r and r0 according to mode
        if self.is_diverge():
            raise Exception("diverge")
        if self.is_converge():
            return True
        return False

    def is_diverge(self):
        if np.isnan(self.r) or np.isinf(self.r):
            return True
        return False

    def is_converge(self):
        if self.r<self.tol:
            logging.info(f"converge by atol {self.r:.2e} < {self.tol:.2e}")
            return True
        if self.r<self.rtol * self.r0:
            logging.info(f"converge by rtol {self.r:.2e} < {self.rtol * self.r0:.2e}")
            return True
        return False

    def calc_r(self, frame, ite, tic_iter, r_Axb=None):
        self.tic_calcr = perf_counter()
        self.t_iter = perf_counter()-tic_iter

        self.frame = frame
        self.ite = ite
        s = f"{frame}-{ite} "
        
        if self.use_calc_dual:
            self.dual = self.calc_dual()
            s += f"dual0:{self.dual0:.2e} "
            s += f"dual:{self.dual:.2e} "

        if r_Axb is not None:
            self.r_Axb = r_Axb.tolist() if not isinstance(r_Axb, list) else r_Axb
            self.ninner = len(r_Axb)-1
            s += f" rsys:{self.r_Axb[0]:.2e} {self.r_Axb[-1]:.2e} "
            s += f"ninner:{self.ninner} "
            self.conv_factor = calc_conv(r_Axb)
            s += f"conv:{self.conv_factor:.2f} "

        if self.use_calc_primal and self.calc_primal is not None:
            self.primal, self.Newton = self.calc_primal()
            s += f"Newton:{self.Newton:.2e} primal:{self.primal:.2e} "

        if self.use_calc_energy and self.calc_energy is not None:
            self.energy = self.calc_energy()
            s += f"energy:{self.energy:.8e} "

        if self.use_calc_strain and self.calc_strain is not None:
            self.max_strain = self.calc_strain()
            s += f"strain:{self.max_strain:.2e} "

        self.t_calcr = perf_counter()-self.tic_calcr
        s += f"calcr:{self.t_calcr*1000:.2f}ms "

        self.t_export += self.t_calcr

        s+= f" t_iter:{self.t_iter*1000:.2f}ms"

        logging.info(s)

    def calc_r0(self):
        tic = perf_counter()
        self.dual0 = self.calc_dual()
        if self.use_calc_primal:
            self.primal0, self.Newton0 = self.calc_primal()
        if self.use_calc_energy:
            self.energy0 = self.calc_energy()

        self.t_export = perf_counter()-tic # reset t_export here

    def choose_mode(self, converge_condition):
        if converge_condition == "dual":
            self.mode = ResidualType.dual
        elif converge_condition == "primal":
            self.mode = ResidualType.primal
        elif converge_condition == "energy":
            self.mode = ResidualType.energy
        elif converge_condition == "strain":
            self.mode = ResidualType.strain
        elif converge_condition == "Newton":
            self.mode = ResidualType.Newton

    def set_r(self):
        if self.mode == ResidualType.dual:
            self.r = self.dual
            self.r0 = self.dual0
        elif self.mode == ResidualType.Newton:
            self.r = self.Newton
            self.r0 = self.Newton0
        elif self.mode == ResidualType.primal:
            self.r = self.primal
            self.r0 = self.primal0
        elif self.mode == ResidualType.energy:
            self.r = self.energy
            self.r0 = self.energy0
        elif self.mode == ResidualType.strain:
            self.r = self.max_strain
            self.r0 = self.max_strain0


def calc_conv(r):
    return (r[-1]/r[0])**(1.0/(len(r)-1))

@ti.kernel
def calc_norm(a:ti.template())->ti.f32:
    sum = 0.0
    for i in range(a.shape[0]):
        sum += a[i] * a[i]
    sum = ti.sqrt(sum)
    return sum

@dataclass
class ResidualDataOneFrame:
    name = "residualOneFrame"
    r_iters: list # list of ResidualDataOneIter NOT USED
    n_outer: int=0
    frame: int=0
    t: float=0
    # t_export: float=0


@dataclass
class ResidualDataAllFrame:
    name = "residualAll"
    r_frames: list # list of ResidualDataOneFrame NOT USED
    stalled_frame: list
    t: float=0
    t_export: float=0


def ending(args, ist):
    import time, datetime
    from pathlib import Path

    t_all = time.perf_counter() - ist.timer_loop
    end_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.end_frame = ist.frame

    len_n_outer_all = len(ist.n_outer_all) if len(ist.n_outer_all) > 0 else 1
    sum_n_outer = sum(ist.n_outer_all)
    avg_n_outer = sum_n_outer / len_n_outer_all
    max_n_outer = max(ist.n_outer_all) if ist.n_outer_all else 0
    max_n_outer_index = ist.n_outer_all.index(max_n_outer) if ist.n_outer_all else 0

    n_outer_all_np = np.array(ist.n_outer_all, np.int32)    
    np.savetxt(args.out_dir+"/r/n_outer.txt", n_outer_all_np, fmt="%d")

    sim_time_with_export = time.perf_counter() - ist.timer_loop
    sim_time = sim_time_with_export - ist.r_all.t_export
    nframes = (args.end_frame - ist.initial_frame) if args.end_frame > ist.initial_frame else 1
    avg_sim_time = sim_time / nframes

    s = f"\n-------\n"+\
    f"Time: {(sim_time):.2f}s = {(sim_time)/60:.2f}min.\n" + \
    f"Time with exporting: {(sim_time_with_export):.2f}s = {sim_time_with_export/60:.2f}min.\n" + \
    f"Time of exporting: {ist.r_all.t_export:.3f}s\n" + \
    f"Frame {ist.initial_frame}-{args.end_frame}({args.end_frame-ist.initial_frame} frames)."+\
    f"\nAvg: {avg_sim_time}s/frame."+\
    f"\nStart\t{ist.start_date},\nEnd\t{end_date}."+\
    f"\nSum n_outer: {sum_n_outer} \nAvg n_outer: {avg_n_outer:.1f}"+\
    f"\nMax n_outer: {max_n_outer} \nMax n_outer frame: {max_n_outer_index + ist.initial_frame}." + \
    f"\nstalled at {ist.all_stalled}"+\
    f"\n{ist.sim_name}" + \
    f"\ndt={args.delta_t}" + \
    f"\nSolver: {args.solver_type}" + \
    f"\nout_dir: {args.out_dir}" 

    logging.info(s)

    out_dir_name = Path(args.out_dir).name
    name = ist.start_date + "_" +  str(out_dir_name) 
    file_name = f"result/meta/{name}.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(s)

    file_name2 = f"{args.out_dir}/meta.txt"
    with open(file_name2, "w", encoding="utf-8") as file:
        file.write(s)

    if args.solver_type == "AMGX":
        ist.linsol.finalize()
    exit()


def export_after_substep(ist, args, **kwargs):
    import time
    from engine.mesh_io import write_mesh, write_edge_data, write_ply_with_strain, edge_data_to_tri_data, write_vtk_with_strain
    from engine.file_utils import save_state
    tic_export = time.perf_counter()
    if args.export_mesh and not args.use_houdini_data:
        pos_np = ist.pos.to_numpy() if type(ist.pos) != np.ndarray else ist.pos
        write_mesh(args.out_dir + f"/mesh/{ist.frame:04d}", pos_np, ist.tri)
        if args.export_strain:
            if ist.sim_type == "cloth":
                # v1: simply write txt, need post process
                # write_edge_data(args.out_dir + f"/mesh/{ist.frame:04d}_strain", ist.strain.to_numpy())
                # v2: write mesh with strain directly in simulation
                tri = ist.tri
                ist.strain_cell = edge_data_to_tri_data(ist.e2t, ist.strain.to_numpy(), tri)
                write_ply_with_strain(args.out_dir + f"/mesh/{ist.frame:04d}", pos_np, tri, strain=ist.strain_cell, binary=True)
    if args.export_mesh and args.use_houdini_data:
        ist.write_geo(args.out_dir + f"/mesh/{ist.frame:04d}.geo")
    
    if args.export_state:
        save_state(args.out_dir+'/state/' + f"{ist.frame:04d}.npz", ist)
    ist.r_all.t_export += time.perf_counter()-tic_export
    t_frame = time.perf_counter()-ist.tic_frame
    if args.export_log:
        logging.info(f"Time of frame-{ist.frame}: {t_frame:.3f}s")


def init_logger(args):
    import sys
    log_level = logging.INFO
    if not args.export_log:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format="%(message)s",filename=args.out_dir + f'/latest.log',filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(args)


def report_multilevel_details(Ps, num_levels):
    logging.info(f"    num_levels:{num_levels}")
    num_points_level = []
    for i in range(len(Ps)):
        num_points_level.append(Ps[i].shape[0])
    num_points_level.append(Ps[-1].shape[1])
    for i in range(num_levels):
        logging.info(f"    num points of level {i}: {num_points_level[i]}")


def do_export_r(r, out_dir, frame):
    tic = perf_counter()
    serialized_r = [r[i].__dict__ for i in range(len(r))]
    r_json = json.dumps(serialized_r)
    with open(out_dir+'/r/'+ f'{frame}.json', 'w') as file:
        file.write(r_json)
    r.t_export += perf_counter()-tic


def export_mat(ist,get_A,b):
    args = ist.args
    tic = perf_counter()
    if not args.export_matrix or get_A is None:
        return
    if ist.frame != args.export_matrix_frame:
        return
    if hasattr(args, "export_matrix_ite"):
        if ist.ite != args.export_matrix_ite:
            return
    else:
        if ist.ite != 0:
            return
    if args.export_matrix_dir is None:
        dir = args.out_dir + "/A/"
    else:
        dir = args.export_matrix_dir
    A = get_A()
    postfix = f"F{ist.frame}" if ist.frame is not None else ""
    export_A_b(A, b, dir=dir, postfix=postfix, binary=args.export_matrix_binary)
    ist.r_iter.t_export += perf_counter()-tic


def export_A_b(A, b, dir, postfix=f"", binary=True):
    from time import perf_counter
    import scipy

    print(f"Exporting A and b to {dir} with postfix {postfix}")
    tic = perf_counter()
    if binary:
        # https://stackoverflow.com/a/8980156/19253199
        scipy.sparse.save_npz(dir + f"/A_{postfix}.npz", A)
        if b is not None:
            np.save(dir + f"/b_{postfix}.npy", b)
        # A = scipy.sparse.load_npz("A.npz") # load
        # b = np.load("b.npy")
    else:
        scipy.io.mmwrite(dir + f"/A_{postfix}.mtx", A, symmetry='symmetric')
        if b is not None:
            np.savetxt(dir + f"/b_{postfix}.txt", b)
    print(f"    export_A_b time: {perf_counter()-tic:.3f}s")


def do_post_iter(ist, get_A0_cuda=None):
    ist.update_constraints() #CAUTION that this should be called before calc_r
    ist.r_iter.calc_r(ist.frame,ist.ite, ist.r_iter.tic_iter, ist.r_iter.r_Axb)
    export_mat(ist, get_A0_cuda, ist.b)
    ist.r_all.t_export += ist.r_iter.t_export
    ist.r_iter.t_export = 0.0
    logging.info(f"iter time(with export): {(perf_counter()-ist.r_iter.tic_iter)*1000:.0f}ms")





def main_loop(ist,args):
    import time
    import tqdm

    ist.timer_loop = time.perf_counter()
    ist.initial_frame = ist.frame
    step_pbar = tqdm.tqdm(total=args.end_frame, initial=ist.frame)
    ist.r_all.t_export = 0.0

    try:
        for f in range(ist.initial_frame, args.end_frame):
            ist.tic_frame = time.perf_counter()

            if args.solver_type == "XPBD":
                ist.substep_xpbd()
            elif args.solver_type == "NEWTON":
                ist.substep_newton()
            else:
                ist.substep_all_solver()

            export_after_substep(ist,args)
            ist.frame += 1

            logging.info("\n")
            step_pbar.update(1)
            logging.info("\n")
            
        print("Normallly end.")
        ending(args,ist)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        ending(args,ist)


def timeit(method):
    def timed(*args, **kw):
        ts = perf_counter()
        result = method(*args, **kw)
        te = perf_counter()
        logging.info(f"    {method.__name__} took: {(te-ts)*1000:.0f}ms")
        return result
    return timed


def norm_sqr(x):
    return np.linalg.norm(x)**2

def norm(x):
    return np.linalg.norm(x)

def normalize(x):
    return x / np.linalg.norm(x)


def spy_A(A,b):
    import scipy.io
    import matplotlib.pyplot as plt
    print("A:", A.shape, " b:", b.shape)
    scipy.io.mmwrite("A.mtx", A)
    plt.spy(A, markersize=1)
    plt.show()
    exit()


def is_symmetric(A):
    AT = A.transpose()
    diff = A - AT
    if diff.nnz == 0:
        return True
    maxdiff = np.max(np.abs(diff.data))
    return maxdiff < 1e-6

def csr_is_equal(A, B, tol=1e-4):
    if A.shape != B.shape:
        print("shape not equal")
        assert False
    diff = A - B
    if diff.nnz == 0:
        print("csr is equal! nnz=0")
        return True
    maxdiff = np.abs(diff.data).max()
    where = np.abs(diff.data).argmax()
    coo = A.tocoo()
    i,j = coo.row[where], coo.col[where]
    print("maxdiff: ", maxdiff)
    if maxdiff > tol:
        assert False, f"maxdiff:{maxdiff}, where=({i},{j})"
    print("csr is equal!")
    return True

def dense_mat_is_equal(A, B):
    diff = A - B
    maxdiff = np.abs(diff).max()
    print("maxdiff: ", maxdiff)
    if maxdiff > 1e-6:
        assert False
    print("is equal!")
    return True

def vec_is_equal(a, b, maxdiff=1e-6):
    if a.shape[-1]==1:
        a = a.reshape(-1)
    if b.shape[-1]==1:
        b = b.reshape(-1)
    if a.shape != b.shape:
        assert False, f"shape not equal: {a.shape} != {b.shape}"
    diff = a - b
    maxdiff = np.abs(diff).max()
    where = np.abs(diff.data).argmax()
    print("maxdiff: ", maxdiff)
    if maxdiff > 1e-6:
        assert False, f"maxdiff:{maxdiff}, where={where}"
    print("is equal!")
    return True


def debug(x, name='vec'):  
    print(f'{name}: {x.shape}')
    norm = np.linalg.norm(x)
    max_val = np.max(x)
    amax = np.argmax(x)
    min_val = np.min(x)
    amin = np.argmin(x)
    print(f'    norm: {norm} max_val: {max_val}, amax: {amax} min_val: {min_val}, amin: {amin}\n')
    np.savetxt(f'{name}.txt', x)

def debugmat(x, name='mat'):  
    print(f'{name}: {x.shape}')
    norm = np.linalg.norm(x.data)
    max_val = np.max(x.data)
    min_val = np.min(x.data)
    print(f'    norm: {norm} max_val: {max_val}  min_val: {min_val}\n')
    scipy.io.mmwrite(f"{name}.mtx", x)


def set_mass_matrix(mass:np.ndarray):
    mass3 = np.repeat(mass, 3)
    MASS = scipy.sparse.diags(mass3, 0)
    return MASS

def set_mass_matrix_from_invmass(inv_mass:np.ndarray):
    if type(inv_mass) != np.ndarray:
        inv_mass_np = inv_mass.to_numpy()
    else:
        inv_mass_np = inv_mass
    where_zeros = np.where(inv_mass_np==0)
    mass = 1.0/inv_mass_np
    mass[where_zeros] = 0
    mass3 = np.repeat(mass, 3)
    MASS = scipy.sparse.diags(mass3)
    return MASS

def set_inv_mass_from_mass_matrix(inv_mass, MASS):
    mass3 = MASS.diagonal()
    mass = mass3[::3]
    inv_mass_np = 1.0/mass
    inv_mass.from_numpy(inv_mass_np)
    return inv_mass

def eliminate_zero_inv_mass(A, b, inv_mass):
    if type(inv_mass) != np.ndarray:
        inv_mass_np = inv_mass.to_numpy()
    else:
        inv_mass_np = inv_mass
    where = np.where(inv_mass_np==0)
    logging.info(f"Original shape of A:{A.shape}")
    logging.info(f"Deleting the row and col {where[0]} from A and b")
    b = np.delete(b,where)
    wherel = where[0].tolist()
    A = delete_from_csr(A,wherel, wherel)
    logging.info(f"Current shape of A:{A.shape}")
    return A,b

def find_zero_inv_mass(inv_mass):
    if type(inv_mass) != np.ndarray:
        inv_mass_np = inv_mass.to_numpy()
    else:
        inv_mass_np = inv_mass
    where = np.where(inv_mass_np==0)
    return  where



# https://stackoverflow.com/a/45486349/19253199
def delete_from_csr(mat, row_indices=[], col_indices=[]):
    from scipy.sparse import csr_matrix
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat



def set_gravity_as_force(mass, gravity=[0,-9.8,0]):
    assert len(gravity) == 3, "gravity should be a 3d vector"
    NV = mass.shape[0]
    mass3 = np.repeat(mass, 3)
    gravity_constant = np.array(gravity)
    external_acc = np.tile(gravity_constant, NV)
    force = mass3 * external_acc
    return force.reshape(-1,3)

def pinlist_to_field(pinlist, pinposlist,NV):
    pin = ti.field(dtype=ti.i8, shape=NV)
    pinpos = ti.Vector.field(3, dtype=ti.f32, shape=NV)

    for i, v in enumerate(pinlist):
        pin[pinlist[i]] = 1
        pinpos[pinlist[i]] = pinposlist[i]

    return pin, pinpos

def pinlist_to_np(pinlist, pinposlist,NV):
    # pin = ti.field(dtype=ti.i8, shape=NV)
    # pinpos = ti.Vector.field(3, dtype=ti.f32, shape=NV)
    pin = np.zeros(NV, np.int8)
    pinpos = np.zeros((NV,3), np.float32)

    for i, v in enumerate(pinlist):
        pin[pinlist[i]] = 1
        pinpos[pinlist[i]] = pinposlist[i]

    return pin, pinpos



def python_list_to_ti_field(l:list):
    t = type(l[0])
    if t == int:
        ndarr = np.array(l, dtype = t)
        f = ti.field(dtype=ti.i32, shape=len(l))
    elif t == float:
        ndarr = np.array(l, dtype = t)
        f = ti.field(dtype=ti.f32, shape=len(l))
    elif t == bool:
        ndarr = np.array(l, dtype = t)
        f = ti.field(dtype=ti.i8, shape=len(l))
    elif t == list:
        tt = type(l[0][0])
        ndarr = np.array(l, dtype = tt)
        f = ti.Vector.field(len(l[0]), dtype=tt, shape=len(l))
    f.from_numpy(ndarr)
    return f
        