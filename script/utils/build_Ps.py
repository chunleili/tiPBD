import pyamg

def build_Ps(A, method='UA'):
    """Build a list of prolongation matrices Ps from A """
    if method == 'SA' or 'SA_CG':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400)
    elif method == 'UA' or 'UA_CG':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None)
    elif method == 'UA_CG_GS':
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, smooth=None, coarse_solver='gauss_seidel')
    elif method == 'CAMG' or 'CAMG_CG':
        ml = pyamg.ruge_stuben_solver(A, max_coarse=400)
    else:
        raise ValueError(f"Method {method} not recognized")

    Ps = []
    for i in range(len(ml.levels)-1):
        Ps.append(ml.levels[i].P)

    return Ps