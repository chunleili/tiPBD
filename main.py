import os
import ui.argparser
from  engine.solver import Solver

root_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(root_path, "result")

if __name__ == "__main__":
    print("root_path: ",root_path)

    solver = Solver()
    solver.run()
    pass