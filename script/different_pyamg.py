def CR():
    import numpy as np
    import pyamg
    import matplotlib.pyplot as plt

    n = 20
    A = pyamg.gallery.poisson((n,n)).tocsr()

    xx = np.linspace(0,1,n)
    x,y = np.meshgrid(xx,xx)
    V = np.concatenate([[x.ravel()],[y.ravel()]],axis=0).T

    splitting = pyamg.classical.cr.CR(A)

    C = np.where(splitting == 0)[0]
    F = np.where(splitting == 1)[0]

    fig, ax = plt.subplots()
    ax.scatter(V[C, 0], V[C, 1], marker='s', s=18,
            color=[232.0/255, 74.0/255, 39.0/255], label='C-pts')
    ax.scatter(V[F, 0], V[F, 1], marker='s', s=18,
            color=[19.0/255, 41.0/255, 75.0/255], label='F-pts')
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
            borderaxespad=0, ncol=2)

    ax.axis('square')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    figname = './output/crsplitting.png'
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--savefig':
            plt.savefig(figname, bbox_inches='tight', dpi=150)
    else:
        plt.show()

def load_A_b(case_name = "scale64"):
    import scipy.io
    import os
    import numpy as np
    postfix = "F1-0"

    print("loading data...")
    prj_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
    to_read_dir = prj_dir + f"result/{case_name}/A/"
    A = scipy.io.mmread(to_read_dir+f"A_{postfix}.mtx")
    A = A.tocsr()
    print("shape of A: ", A.shape)
    b = np.loadtxt(to_read_dir+f"b_{postfix}.txt", dtype=np.float64)

    return A, b

def strength():
    import numpy as np
    import pyamg
    import matplotlib.pyplot as plt
    import time

    # n = int(1e2)
    # stencil = pyamg.gallery.diffusion_stencil_2d(type='FE', epsilon=0.001, theta=np.pi / 3)
    # A = pyamg.gallery.stencil_grid(stencil, (n, n), format='csr')
    # b = np.random.rand(A.shape[0])
    case_name = "scale64"
    A,b = load_A_b(case_name)
    x0 = 0 * b

    runs = []
    options = []
    options.append(('symmetric', {'theta': 0.0}))
    options.append(('symmetric', {'theta': 0.25}))
    options.append(('evolution', {'epsilon': 4.0}))
    options.append(('affinity', {'epsilon': 3.0, 'R': 10, 'alpha': 0.5, 'k': 20}))
    options.append(('affinity', {'epsilon': 4.0, 'R': 10, 'alpha': 0.5, 'k': 20}))
    options.append(('algebraic_distance',
                {'epsilon': 2.0, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))
    options.append(('algebraic_distance',
                {'epsilon': 3.0, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))

    for opt in options:
        #optstr = opt[0] + '\n    ' + \
        #    ',\n    '.join(['%s=%s' % (u, v) for (u, v) in list(opt[1].items())])
        optstr = opt[0] + ': ' + \
            ', '.join(['%s=%s' % (u, v) for (u, v) in list(opt[1].items())])
        print("running %s" % (optstr))

        tic = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(
            A,
            strength=opt,
            max_levels=10,
            max_coarse=5,
            keep=False)
        res = []
        x = ml.solve(b, x0, tol=1e-12, residuals=res)
        runs.append((res, optstr))
        print(f"Elapsed time: {time.perf_counter() - tic:0.4f} seconds")

    fig, ax = plt.subplots()
    for run in runs:
        label = run[1]
        label = label.replace('theta', '$\\theta$')
        label = label.replace('epsilon', '$\\epsilon$')
        label = label.replace('alpha', '$\\alpha$')
        ax.semilogy(run[0], label=label, linewidth=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Residual')

    #l4 = plt.legend(bbox_to_anchor=(0,1.02,1,0.5), loc="lower left",
    #                mode="expand", borderaxespad=0, ncol=1)
    plt.legend(loc="lower left", borderaxespad=0, ncol=1, frameon=False)
    plt.title(f'{case_name}: Strength Options')

    figname = f'./output/strength_options.png'
    import sys
    if '--savefig' in sys.argv:
        plt.savefig(figname, bbox_inches='tight', dpi=150)
    else:
        plt.show()


if __name__ == '__main__':
    strength()