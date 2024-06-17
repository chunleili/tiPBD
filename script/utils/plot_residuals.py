def plot_residuals_all(allres,show_fig=True,save_fig=True,plot_title='Residuals'):

    import matplotlib.pyplot as plt
    import os
    from utils.mkdir_if_not_exist import mkdir_if_not_exist
    from utils.define_to_read_dir import to_read_dir

    # draw_plot
    colors = ['blue', 'orange', 'red', 'purple', 'green', 'black', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'brown', 'pink']
    markers = ['o', 'x', 's', 'd', '^', 'v', '>', '<', '1', '2', '3', '4', '+', 'X']

    # https://matplotlib.org/stable/api/markers_api.html for different markers
    # https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def for different colors
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    fig, axs = plt.subplots(1, figsize=(8, 9))
    
    def a2r(r): #absolute to relative
        return r/r[0]
    
    for i in range(len(allres)):
        # if allres[i].label == 'SA+CG' or\
        #    allres[i].label == 'UA+CG' or\
        #    allres[i].label == 'GS':
        plot_residuals(a2r(allres[i].r), axs,  label=allres[i].label)

    fig.canvas.manager.set_window_title(plot_title)
    plt.tight_layout()
    if save_fig:
        dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
        mkdir_if_not_exist(dir)
        plt.savefig(dir+f"/residuals_{plot_title}.png")
    if show_fig:
        plt.show()



def plot_residuals(data, ax, *args, **kwargs):
    import numpy as np
    title = kwargs.pop("title", "")
    linestyle = kwargs.pop("linestyle", "-")
    label = kwargs.pop("label", "")
    x = np.arange(len(data))
    ax.plot(x, data, label=label, linestyle=linestyle, *args, **kwargs)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative residual")
    ax.legend(loc="upper right")