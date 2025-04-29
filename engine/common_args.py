def add_common_args(parser):
    parser.add_argument("-maxiter", type=int, default=10)
    parser.add_argument("-delta_t", type=float, default=3e-3)
    parser.add_argument("-solver_type", type=str, default="AMG", choices=["XPBD",  "AMG", "AMGX", "DIRECT", "LUMPED", "GS","NEWTON"])
    parser.add_argument("-linsol_type", type=str, default="AMG", choices=["AMG", "AMGX", "DIRECT", "GS"])
    parser.add_argument("-end_frame", type=int, default=10)
    parser.add_argument("-out_dir", type=str, default="result/latest/")
    parser.add_argument("-export_matrix", type=int, default=False)
    parser.add_argument("-export_matrix_binary", type=int, default=True)
    parser.add_argument("-export_matrix_dir", type=str, default=None)
    parser.add_argument("-export_matrix_frame", type=int, default=None)
    parser.add_argument("-auto_another_outdir", type=int, default=False)
    parser.add_argument("-use_cuda", type=int, default=True)
    parser.add_argument("-cuda_dir", type=str, default="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")
    parser.add_argument("-amgx_lib_dir", type=str, default="D:/Dev/AMGX/build/Release")
    parser.add_argument("-build_P_method", type=str, default="UA")
    parser.add_argument("-arch", type=str, default="cpu")
    parser.add_argument("-setup_interval", type=int, default=10000)
    parser.add_argument("-maxiter_Axb", type=int, default=100)
    parser.add_argument("-export_log", type=int, default=True)
    parser.add_argument("-export_residual", type=int, default=False)
    parser.add_argument("-restart", type=int, default=False)
    parser.add_argument("-restart_file", type=str, default="result/latest/state/20.npz")
    parser.add_argument("-use_cache", type=int, default=False)
    parser.add_argument("-export_mesh", type=int, default=True)
    parser.add_argument("-tol", type=float, default=1e-4)
    parser.add_argument("-atol", type=float, default=1e-4, help="absolute tolerance, same with tol")
    parser.add_argument("-rtol", type=float, default=1e-9)
    parser.add_argument("-tol_Axb", type=float, default=1e-5)
    parser.add_argument("-smoother_niter", type=int, default=2)
    parser.add_argument("-filter_P", type=str, default=None)
    parser.add_argument("-scale_RAP", type=int, default=False)
    parser.add_argument("-only_smoother", type=int, default=False)
    parser.add_argument("-only_PCG", type=int, default=False)
    parser.add_argument("-debug", type=int, default=False)
    parser.add_argument("-coarse_solver_type", type=int, default=1, help="0: direct solver, 1: smoother")
    parser.add_argument("-amgx_config", type=str, default="data/config/FGMRES_CLASSICAL_AGGRESSIVE_PMIS.json")
    parser.add_argument("-export_state", type=int, default=False)
    parser.add_argument("-use_json", type=int, default=False, help="json configs will overwrite the command line args")
    parser.add_argument("-json_path", type=str, default="", help="json configs will overwrite the command line args")
    parser.add_argument("-yaml_path", type=str, default="", help="yaml configs")
    parser.add_argument("-gravity", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("-use_gravity", type=int, default=True)
    parser.add_argument("-converge_condition", type=str, default="dual", choices=["dual", "Newton", "strain", "time"], help="dual: dual residual, Newton: sqrt(dual^2+primal^2), strain: strain limiting, time: time budget")
    parser.add_argument("-use_withK", type=int, default=False)
    parser.add_argument("-export_strain", type=int, default=False)
    parser.add_argument("-calc_dual", type=int, default=True)
    parser.add_argument("-calc_strain", type=int, default=False)
    parser.add_argument("-calc_energy", type=int, default=False)
    parser.add_argument("-calc_primal", type=int, default=False)
    parser.add_argument("-use_pintoanimation", type=int, default=False)
    parser.add_argument("-use_houdini_data", type=int, default=False)
    parser.add_argument("-use_ground_collision", type=int, default=False)
    parser.add_argument("-geo_dir", type=str, default=f"data/model/extraSpring/")
    parser.add_argument("-use_extra_spring", type=int, default=False)
    parser.add_argument("-use_external_constraints", type=int, default=False)
    parser.add_argument("-use_pintotarget", type=int, default=False)
    parser.add_argument("-use_muscle2muscle", type=int, default=False)
    parser.add_argument("-start_frame", type=int, default=1)
    parser.add_argument("-clean_dir", type=int, default=False)
    parser.add_argument("-export_fulldual", type=int, default=False)
    parser.add_argument("-time_budget", type=float, default=1000.0)
    parser.add_argument("-use_time_budget", type=int, default=False)
    parser.add_argument("-calc_rbm", type=int, default=False)
    parser.add_argument("-local_interval", type=int, default=0)
    parser.add_argument("-pmass", type=float, default=1.0)
    parser.add_argument("-use_totalmass", type=int, default=0)
    parser.add_argument("-total_mass", type=float, default=-1)#16000.0
    parser.add_argument("-mass_density", type=float, default=-1)#1000.0
    parser.add_argument("-direct_solver_type", type=str, default="cusolver", choices= ["cusolver", "scipy", "pardiso"])
    parser.add_argument("-fixed_stiffness", type=float, default=1e8)
    parser.add_argument("-fixed_particles", type=int, nargs="*", default=[])
    parser.add_argument("-verbosity", type=int, default=1)
    parser.add_argument("-use_WangChebyshev", type=int, default=False)
    parser.add_argument("-WangChebyshev_rho", type=float, default=0.9992)
    parser.add_argument("-WangChebyshev_gamma", type=float, default=0.9)
    parser.add_argument("-WangChebyshev_S", type=int, default=9)
    parser.add_argument("-rescale_to_unit_cube", type=int, default=False)
    parser.add_argument("-ground_pos", type=float, default=0.0)
    parser.add_argument("-use_SDF_collision", type=int, default=False, help="a master switch to use SDF collision, if false, all SDF collision will not be used")
    parser.add_argument("-collider_json_path", type=str, default="", help="json file for colliders, see data/scene/sphere.json for example")
    parser.add_argument("-visualize_colliders", type=int, default=True, help="A master swich to visualize colliders, if false, all colliders will not be visualized even if they are set to be visible")
    parser.add_argument("-collision_nsubsteps", type=int, default=1, help="number of substeps for collision response")
    parser.add_argument("-initial_translate", type=float, nargs=3, default=(0.0, 0.0, 0.0),help="initial_translate for model")
    return parser


def parse_json_args(args,json_path=""):
    if not args.use_json:
        return
    import json
    import os
    if not os.path.exists(json_path):
        assert False, f"json file {json_path} not exist!"
    print(f"CAUTION: using json config file {json_path} to overwrite the command line args!")
    if json_path=="" and os.path.exists("config"):
        print("Using config file  to set json path")
        with open("config") as f:
            json_path = f.read().strip()
            args.json_path = json_path
    else:
        Warning("No json")
    print(f"use json_path: {json_path}")
    with open(json_path, "r") as json_file:
        config = json.load(json_file)
    for key, value in config.items():
        if hasattr(args,key):
            if getattr(args,key) != value:
                # print(f"overwriting {key} from {getattr(args,key)} to {value}")
                setattr(args,key,value)
        else:
            # print(f"Add new json key {key}:{value} to args")
            setattr(args,key,value)
    return args
