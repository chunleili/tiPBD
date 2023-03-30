from  engine.solver import Solver
import taichi as ti
from ui.parse_commandline_args import parse_commandline_args
if __name__ == "__main__":
    args = parse_commandline_args()
    ti.init(arch=args.arch, kernel_profiler=True)

    solver = Solver()
    solver.run()