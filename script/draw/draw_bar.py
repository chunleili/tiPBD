import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from brokenaxes import brokenaxes

def draw_times(times, labels):
    plt.rcParams.update({'font.size': 15})


    assert len(times) == len(labels)
    print("\n\nTime(s) taken for each solver")
    for i in range(len(labels)):
        print(f"{labels[i]}:\t{times[i]:.2f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # bax = brokenaxes(ylims=((0, 4), (91, 93)), hspace=.05, despine=False)

    # 去掉颜色
    # bars = ax.barh(range(len(times)), times, color='white', edgecolor='black')
    colors = ['#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']
    ax.bar(range(len(times)), times,  edgecolor='black', color=colors)

    ax.set_ylabel('Time(s)', fontsize=15)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(labels, fontsize=15)  # 增大字体大小
    ax.tick_params(axis='x', labelsize=15)
    # 调整 y 轴标签的位置
    ax.set_xticklabels(labels, fontsize=15)
    fig.tight_layout()
    plt.savefig("time.png", dpi=300)
    # ax.set_title("Time taken for each solver", fontsize=15)  # 增大字体大小





def broken_bars(times, labels):
    import matplotlib.pyplot as plt
    import numpy as np


    # font = {'family' : 'normal',
    #         'weight' : 'bold',
    #         'size'   : 22}

    # matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': 15})


    # np.random.seed(19680801)

    # pts = np.random.rand(30)*.2
    pts=times
    # Now let's make two outlier points which are far away from everything.
    # pts[[3, 14]] += .8

    # If we were to simply plot pts, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax1) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

    # plot the same data on both Axes
    # ax1.plot(pts)
    # ax2.plot(pts)
    colors = ['#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']
    ax1.bar(labels, times,  edgecolor='black', color=colors)
    ax2.bar(labels, times,  edgecolor='black',color=colors)

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(80,95)  # outliers only
    ax2.set_ylim(0, 5)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    # 减少tick的数量
    # ax1.locator_params(axis='y', major_locator=4)
    # ax2.locator_params(axis='y', major_locator=4)
    # 只用主刻度
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))

    # ax1.set_yticks([90, 91, 92, 93])
    # ax2.set_yticks([0, 1, 2, 3, 4])
    # ax1.set_xticks(range(len(times)), fontsize=15)
    # ax2.set_xticks(range(len(times)), fontsize=15)
    ax1.set_xticklabels(labels,fontsize=15)
    ax2.set_xticklabels(labels,fontsize=15)
    # # ax1.set_yticklabels(times, fontsize=15)
    # # ax2.set_yticklabels(times, fontsize=15)
    ax1.set_ylabel('Time(s)', fontsize=15)
    # ax2.set_ylabel('Time(s)', fontsize=15)
    ax1.yaxis.set_label_coords(-0.1, 0)

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the Axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the Axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)



    plt.show()





if __name__ == "__main__":
    labels = ["AMGX", "AMGCL", "Ours", "Pardiso" ]
    times = [3.85, 1.63,  0.54, 91.98]

    # draw_times(times, labels)
    broken_bars(times, labels)
    plt.show()


    # import matplotlib.pyplot as plt
    # from brokenaxes import brokenaxes
    # import numpy as np

    # fig = plt.figure(figsize=(5,2))
    # bax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05, despine=False)
    # x = np.linspace(0, 1, 100)
    # bax.plot(x, np.sin(10 * x), label='sin')
    # bax.plot(x, np.cos(10 * x), label='cos')
    # bax.legend(loc=3)
    # bax.set_xlabel('time')
    # bax.set_ylabel('value')
    # plt.show()
