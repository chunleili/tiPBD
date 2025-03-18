def plot_residuals_all(df,show_fig=True,save_fig=True,postfix='', use_markers=False):
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
        import ast
        res = df.iloc[i].loc['r']
        # res = ast.literal_eval(res)
        label =df.iloc[i].loc['label']

        print(label)
        plot_residuals(a2r(res), axs,  label=label, marker=markers[i], color=colors[i])

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
    ax.plot(x, data, label=label, linestyle=linestyle, linewidth=2, *args, **kwargs)  # 加粗线条
    ax.set_title(title, fontsize=15)  # 增大字体大小
    ax.set_yscale("log")
    ax.set_xlabel("iteration", fontsize=15)  # 增大字体大小
    ax.set_ylabel("relative residual", fontsize=15)  # 增大字体大小
    ax.legend(loc="upper right", fontsize=15)  # 增大字体大小
    ax.tick_params(axis='both', which='major', labelsize=15)  # 加大 tick 的字体




def draw_convergence_factors(convs, labels):
    import matplotlib.pyplot as plt

    assert len(convs) == len(labels)
    print("\n\nConvergence factor of each solver")
    for i in range(len(labels)):
        print(f"{labels[i]}:\t{convs[i]:.3f}")
    fig, ax = plt.subplots()
    ax.barh(range(len(convs)), convs, color='blue')
    ax.set_yticks(range(len(convs)))
    ax.set_yticklabels(labels, fontsize=15)  # 增大字体大小
    ax.set_title("Convergence factor of each solver", fontsize=15)  # 增大字体大小




def draw_times(times, labels):
    import matplotlib.pyplot as plt

    assert len(times) == len(labels)
    print("\n\nTime(s) taken for each solver")
    for i in range(len(labels)):
        print(f"{labels[i]}:\t{times[i]:.2f}")
    fig, ax = plt.subplots()
    ax.barh(range(len(times)), times, color='red')
    ax.set_yticks(range(len(times)))
    ax.set_yticklabels(labels, fontsize=15)  # 增大字体大小
    ax.set_title("Time taken for each solver", fontsize=15)  # 增大字体大小


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
    ax.set_yticklabels(labels, fontsize=15)  # 增大字体大小
    ax.set_title("Time taken for each solver", fontsize=15)  # 增大字体大小

def plot_full_residual(data, title=""):
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    N = np.sqrt(len(data)).astype(int)

    A = np.linspace(1, N, N)
    B = np.linspace(1, N, N)

    X, Y = np.meshgrid(A, B)
    d0 = data[:N*N].reshape((N, N))

    # Plot the surface.
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    surf0 = ax.plot_surface(X, Y, d0, cmap=cm.coolwarm, label="residual0")
    # ax.set_zlim(-.03, .03)
    fig.text(0.5, 0.9, title, ha='center', fontsize=15)  # 增大字体大小
    fig.canvas.manager.set_window_title(title)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf0, shrink=0.5, aspect=5)
