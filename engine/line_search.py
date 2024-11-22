import numpy as np

class LineSearch:
    def __init__(
        self,
        evaluateObjectiveFunction,
        use_line_search=True,
        ls_alpha=0.25,
        ls_beta=0.1,
        ls_step_size=1.0,
        ΕPSILON=1e-15,
    ):
        """
        Args:
            evaluateObjectiveFunction: function to evaluate the objective function, input is x(shape=(NV,3)), output is scalar
            use_line_search: whether to use line search
            ls_alpha: the factor to control the slope
            ls_beta: the factor to decrease the step size
            ls_step_size: the initial step size
            ΕPSILON: the tolerance to determine minimum step size
        
        """
        self.use_line_search = use_line_search
        self.ls_alpha = ls_alpha
        self.ls_beta = ls_beta
        self.ls_step_size = ls_step_size
        self.EPSILON = ΕPSILON
        self.evaluateObjectiveFunction = evaluateObjectiveFunction

    def line_search(self, x, predict_pos, gradient_dir, descent_dir):
        if not self.use_line_search:
            return self.ls_step_size

        t = 1.0/self.ls_beta
        currentObjectiveValue = self.evaluateObjectiveFunction(x, predict_pos)
        ls_times = 0
        while ls_times==0 or (lhs >= rhs and t > self.EPSILON):
            t *= self.ls_beta
            x_plus_tdx = (x.flatten() + t*descent_dir).reshape(-1,3)
            lhs = self.evaluateObjectiveFunction(x_plus_tdx, predict_pos)
            rhs = currentObjectiveValue + self.ls_alpha * t * np.dot(gradient_dir, descent_dir)
            ls_times += 1
        self.energy = lhs
        print(f'    energy: {self.energy}')
        print(f'    ls_times: {ls_times}')

        if t < self.EPSILON:
            t = 0.0
        else:
            self.ls_step_size = t
        return t
    



class LineSearchNaive:
    def __init__(
        self,
        evaluateObjectiveFunction,
        use_line_search=True,
        ls_beta=0.1,
        ls_step_size=1.0,
        ΕPSILON=1e-15,
    ):
        """
        Args:
            evaluateObjectiveFunction: navie version of line search. No need for gradient information, which is equal to alpha=0 of the full version
            use_line_search: whether to use line search
            ls_beta: the factor to decrease the step size
            ls_step_size: the initial step size
            ΕPSILON: the tolerance to determine minimum step size
        
        """
        self.use_line_search = use_line_search
        self.ls_alpha = 0.0
        self.ls_beta = ls_beta
        self.ls_step_size = ls_step_size
        self.EPSILON = ΕPSILON
        self.evaluateObjectiveFunction = evaluateObjectiveFunction

    def line_search(self, x, predict_pos, descent_dir):
        """
        x: position of vertices, shape=(NV,3), numpy array
        predict_pos: predicted position of vertices, shape=(NV,3), numpy array
        descent_dir: dpos, shape=(NV,3), numpy array
        reutrn: step size
        """
        if not self.use_line_search:
            return self.ls_step_size

        t = 1.0/self.ls_beta
        currentObjectiveValue = self.evaluateObjectiveFunction(x, predict_pos)
        ls_times = 0
        while ls_times==0 or (lhs >= rhs and t > self.EPSILON):
            t *= self.ls_beta
            x_plus_tdx = (x.flatten() + t*descent_dir).reshape(-1,3)
            lhs = self.evaluateObjectiveFunction(x_plus_tdx, predict_pos)
            rhs = currentObjectiveValue 
            ls_times += 1
        self.energy = lhs
        print(f'    energy: {self.energy}')
        print(f'    ls_times: {ls_times}')

        if t < self.EPSILON:
            t = 0.0
        else:
            self.ls_step_size = t
        return t
    