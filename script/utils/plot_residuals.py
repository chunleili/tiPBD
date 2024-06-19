def plot_residuals_all(allres,show_fig=True,save_fig=True,postfix='', use_markers=False):
    import matplotlib.pyplot as plt
    import os
    from utils.mkdir_if_not_exist import mkdir_if_not_exist
    from utils.define_to_read_dir import to_read_dir

    # draw_plot
    colors = ['blue', 'orange', 'red', 'purple', 'green', 'black', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'brown', 'pink']
    markers = ['o', 'x', 's', 'd', '^', 'v', '>', '<', '1', '2', '3', '4', '+', 'X']
    if not use_markers:
        markers = [None for _ in range(len(allres))]
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
        plot_residuals(a2r(allres[i].r), axs,  label=allres[i].label, marker=markers[i], color=colors[i])

    fig.canvas.manager.set_window_title(postfix)
    plt.tight_layout()
    if save_fig:
        dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
        mkdir_if_not_exist(dir)
        plt.savefig(dir+f"/residuals_{postfix}.png")
    if show_fig:
        plt.show()


def plot_residuals_all_new(df,show_fig=True,save_fig=True,postfix='', use_markers=False):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from utils.mkdir_if_not_exist import mkdir_if_not_exist
    from utils.define_to_read_dir import to_read_dir

    # draw_plot
    colors = ['blue', 'orange', 'red', 'purple', 'green', 'black', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'brown', 'pink']
    markers = ['o', 'x', 's', 'd', '^', 'v', '>', '<', '1', '2', '3', '4', '+', 'X']
    if not use_markers:
        markers = [None for _ in range(len(df))]
    # https://matplotlib.org/stable/api/markers_api.html for different markers
    # https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def for different colors
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    fig, axs = plt.subplots(1, figsize=(8, 9))
    
    def a2r(r): #absolute to relative
        r = np.array(r)
        return r/r[0]
    
    for i in range(len(df)):
        # if allres[i].label == 'SA+CG' or\
        #    allres[i].label == 'UA+CG' or\
        #    allres[i].label == 'GS':
        plot_residuals(a2r(df.iloc[i].loc['residual']), axs,  label=df.iloc[i].loc['label'], marker=markers[i], color=colors[i])

    fig.canvas.manager.set_window_title(postfix)
    plt.tight_layout()
    if save_fig:
        dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
        mkdir_if_not_exist(dir)
        plt.savefig(dir+f"/residuals_{postfix}.png")
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



def draw_convergence_factors(convs, labels):
    import matplotlib.pyplot as plt

    assert len(convs) == len(labels)
    print("\n\nConvergence factor of each solver")
    for i in range(len(labels)):
        print(f"{labels[i]}:\t{convs[i]:.3f}")
    fig, ax = plt.subplots()
    ax.barh(range(len(convs)), convs, color='blue')
    ax.set_yticks(range(len(convs)))
    ax.set_yticklabels(labels)
    ax.set_title("Convergence factor of each solver")



def draw_times(times, labels):
    import matplotlib.pyplot as plt

    assert len(times) == len(labels)
    print("\n\nTime(s) taken for each solver")
    for i in range(len(labels)):
        print(f"{labels[i]}:\t{times[i]:.2f}")
    fig, ax = plt.subplots()
    ax.barh(range(len(times)), times, color='red')
    ax.set_yticks(range(len(times)))
    ax.set_yticklabels(labels)
    ax.set_title("Time taken for each solver")


def draw_times_new(df):
    import matplotlib.pyplot as plt
    times = df['time'].values
    labels = df['label'].values
    
    assert len(times) == len(labels)
    print("\n\nTime(s) taken for each solver")
    for i in range(len(labels)):
        print(f"{labels[i]}:\t{times[i]:.2f}")
    fig, ax = plt.subplots()
    ax.barh(range(len(times)), times, color='red')
    ax.set_yticks(range(len(times)))
    ax.set_yticklabels(labels)
    ax.set_title("Time taken for each solver")