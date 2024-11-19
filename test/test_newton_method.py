import numpy as np
import os, sys
import scipy
sys.path.append(os.getcwd())
from engine.cloth.newton_method import NewtonMethod
from engine.util import norm

def load_vector(file):
    return scipy.io.mmread(file).toarray().flatten()


class TestNewtonMethod(NewtonMethod):
    def __init__(self,args):
        super().__init__(args)

    def prepare_data(self):
        self.x = load_vector('x.mtx')
        self.gradient_dir = load_vector('gradient_dir.mtx')
        self.descent_dir = load_vector('descent_dir.mtx')
        self.predict_pos = load_vector('predict_pos.mtx')
        print(f"norm of gradient_dir: {norm(self.gradient_dir):.10g}")
        print(f"norm of descent_dir: {norm(self.descent_dir):.10g}")
        print(f"norm of x: {norm(self.x):.10g}")


    def test_obj_function(self):
        self.calc_force()
        print(f"norm of external force: {norm(self.force):.10g}")
        obj = super().evaluateObjectiveFunction(self.x)
        print(f"Objective function: {obj:.10g}")
        return obj
    
    def test_read_constraints(self):
        cons = self.setupConstraints.read_constraints("data/model/fast_mass_spring/constraints.txt")
        self.constraintsNew = cons

    
    def test_evaluateHessian(self, x):
        hessian_py = self.calc_hessian_imply_py(x)
        hessian_ti = self.calc_hessian_imply_ti(x)
        from engine.util import csr_is_equal
        csr_is_equal(hessian_py,hessian_ti)
        return hessian_py
    
    def test_evaluteGradient(self, x):
        gradient_py = self.calc_gradient_imply_py(x)
        gradient_ti = self.calc_gradient_imply_ti(x)
        from engine.util import is_equal
        is_equal(gradient_py, gradient_ti)
        return gradient_py
    

if __name__ == '__main__':
    import argparse
    from engine.common_args import add_common_args
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args()

    t = TestNewtonMethod(args)
    t.prepare_data()
    t.test_read_constraints()
    t.test_obj_function()
    t.test_evaluateHessian(t.x)
    t.test_evaluteGradient(t.x)