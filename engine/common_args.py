def add_common_args(parser):
    parser.add_argument("-maxiter", type=int, default=10)
    parser.add_argument("-delta_t", type=float, default=3e-3)
    parser.add_argument("-solver_type", type=str, default="AMG", choices=["XPBD",  "AMG", "AMGX", "DIRECT", "LUMPED", "GS","NEWTON"])
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
    parser.add_argument("-setup_interval", type=int, default=20)
    parser.add_argument("-maxiter_Axb", type=int, default=100)
    parser.add_argument("-export_log", type=int, default=True)
    parser.add_argument("-export_residual", type=int, default=False)
    parser.add_argument("-restart", type=int, default=False)
    parser.add_argument("-restart_file", type=str, default="result/latest/state/20.npz")
    parser.add_argument("-use_cache", type=int, default=False)
    parser.add_argument("-export_mesh", type=int, default=True)
    parser.add_argument("-tol", type=float, default=1e-4)
    parser.add_argument("-rtol", type=float, default=1e-9)
    parser.add_argument("-tol_Axb", type=float, default=1e-5)
    parser.add_argument("-smoother_niter", type=int, default=2)
    parser.add_argument("-filter_P", type=str, default=None)
    parser.add_argument("-scale_RAP", type=int, default=False)
    parser.add_argument("-only_smoother", type=int, default=False)
    parser.add_argument("-debug", type=int, default=False)
    parser.add_argument("-coarse_solver_type", type=int, default=1, help="0: direct solver, 1: smoother")
    parser.add_argument("-amgx_config", type=str, default="data/config/FGMRES_CLASSICAL_AGGRESSIVE_PMIS.json")
    parser.add_argument("-export_state", type=int, default=False)
    parser.add_argument("-use_json", type=int, default=False, help="json configs will overwrite the command line args")
    parser.add_argument("-json_path", type=str, default="data/scene/cloth/config.json", help="json configs will overwrite the command line args")
    parser.add_argument("-gravity", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("-converge_condition", type=str, default="dual", choices=["dual", "Newton", "strain"], help="dual: dual residual, Newton: sqrt(dual^2+primal^2), strain: strain limiting")
    parser.add_argument("-use_withK", type=int, default=False)
    parser.add_argument("-export_strain", type=int, default=False)
    parser.add_argument("-calc_dual", type=int, default=True)
    parser.add_argument("-calc_strain", type=int, default=False)
    parser.add_argument("-calc_energy", type=int, default=False)
    parser.add_argument("-calc_primal", type=int, default=False)
    parser.add_argument("-use_pintoanimation", type=int, default=False)
    parser.add_argument("-use_ground_collision", type=int, default=False)
    return parser
