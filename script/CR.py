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


def CR():
    import numpy as np
    import pyamg
    import matplotlib.pyplot as plt
    import os
    from reproduce_pyamg import plot_residuals, to_read_dir, mkdir_if_not_exist, save_fig_instad_of_show, show_plot
    save_fig_instad_of_show = False

    A,b = load_A_b()
    maxiter=300

    A.eliminate_zeros()
    ml = pyamg.ruge_stuben_solver(A, CF='CR')
    res = []
    x_pyamg = ml.solve(b, tol=1e-10, residuals=res,maxiter=maxiter)
    print(ml)
    print("res1 classical AMG", len(res), res[-1])
    print((res[-1]/res[0])**(1.0/(len(res)-1)))

    if show_plot:
        fig, axs = plt.subplots(1, figsize=(8, 9))
        plot_residuals(res/res[0], axs,  label="Classical AMG", marker="o", color="blue")
        
        global plot_title
        plot_title = '111'
        fig.canvas.manager.set_window_title(plot_title)
        plt.tight_layout()
        if save_fig_instad_of_show:
            dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
            mkdir_if_not_exist(dir)
            plt.savefig(dir+f"/residuals_{plot_title}.png")
        else:
            plt.show()




if __name__ == '__main__':
    CR()