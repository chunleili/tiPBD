def construct_ml_manually_3levels(A,P0,P1):
    from pyamg.multilevel import MultilevelSolver
    from pyamg.relaxation.smoothing import change_smoothers

    levels = []
    levels.append(MultilevelSolver.Level())
    levels.append(MultilevelSolver.Level())
    levels.append(MultilevelSolver.Level())

    levels[0].A = A
    levels[0].P = P0
    levels[0].R = P0.T
    levels[1].A = P0.T @ A @ P0

    levels[1].P = P1
    levels[1].R = P1.T
    levels[2].A = P1.T @ levels[1].A @ P1

    ml = MultilevelSolver(levels, coarse_solver='pinv')

    presmoother=('block_gauss_seidel',{'sweep': 'symmetric'})
    postsmoother=('block_gauss_seidel',{'sweep': 'symmetric'})
    change_smoothers(ml, presmoother, postsmoother)

    return ml




def construct_ml_manually(A,Ps=[]):
    from pyamg.multilevel import MultilevelSolver
    from pyamg.relaxation.smoothing import change_smoothers

    lvl = len(Ps) + 1 # number of levels

    levels = []
    for i in range(lvl):
        levels.append(MultilevelSolver.Level())

    levels[0].A = A

    for i in range(lvl-1):
        levels[i].P = Ps[i]
        levels[i].R = Ps[i].T
        levels[i+1].A = Ps[i].T @ levels[i].A @ Ps[i]

    ml = MultilevelSolver(levels, coarse_solver='pinv')

    presmoother=('block_gauss_seidel',{'sweep': 'symmetric'})
    postsmoother=('block_gauss_seidel',{'sweep': 'symmetric'})
    change_smoothers(ml, presmoother, postsmoother)

    return ml
