import pyamg

def build_Ps(A, method='UA'):
    """Build a list of prolongation matrices Ps from A """
    if method == 'UA':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, improve_candidates=None, symmetry='symmetric')
    elif method == 'SA' :
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400,symmetry='symmetric')
    elif method == 'CAMG':
        ml = pyamg.ruge_stuben_solver(A, max_coarse=400,symmetry='symmetric')
    elif method == 'adaptive_SA':
        ml = pyamg.aggregation.adaptive_sa_solver(A, max_coarse=400, smooth=None, num_candidates=6)[0]
    elif method == 'rigidbodymodes':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None,symmetry='symmetric')
    else:
        raise ValueError(f"Method {method} not recognized")

    Ps = []
    for i in range(len(ml.levels)-1):
        Ps.append(ml.levels[i].P)

    return Ps