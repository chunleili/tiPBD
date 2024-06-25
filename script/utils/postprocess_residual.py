def calc_conv(r):
    return (r[-1]/r[0])**(1.0/(len(r)-1))


def print_df(labels, convs, times, verbose=False):
    import pandas as pd
    print("\n\nDataframe of convergence factor and time taken for each solver")
    pd.set_option("display.precision", 3)
    df = pd.DataFrame({"label":labels, "conv_fac":convs, "time":times})
    print(df)
    if verbose:
        print("\nIn increasing order of conv_fac:")
        df = df.sort_values(by="conv_fac", ascending=True)
        print(df)
        print("\nIn increasing order of time taken:")
        df = df.sort_values(by="time", ascending=True)
        print(df)
    return df

def print_df_new(df_in, verbose=False):
    import pandas as pd
    print("\n\nDataframe of convergence factor and time taken for each solver")
    pd.set_option("display.precision", 3)
    df = df_in.drop(labels='residual',axis=1)
    print(df)
    if verbose:
        print("\nIn increasing order of conv_fac:")
        df = df.sort_values(by="conv_fac", ascending=True)
        print(df)
        print("\nIn increasing order of time taken:")
        df = df.sort_values(by="time", ascending=True)
        print(df)

def save_data(allres, postfix=""):
    import pandas as pd
    import os
    from script.utils.mkdir_if_not_exist import mkdir_if_not_exist
    from script.utils.define_to_read_dir import to_read_dir

    df = pd.DataFrame(allres)
    dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
    mkdir_if_not_exist(dir)
    df.to_csv(dir+f"/allres_{postfix}.csv")


def save_data_new(df, postfix=""):
    import pandas as pd
    import os
    from script.utils.mkdir_if_not_exist import mkdir_if_not_exist
    from script.utils.define_to_read_dir import to_read_dir

    dir = os.path.dirname(os.path.dirname(to_read_dir)) + '/png/'
    mkdir_if_not_exist(dir)
    df.to_csv(dir+f"/allres_{postfix}.csv")

def postprocess_residual(allres, tic):
    import numpy as np

    # import pandas as pd
    #calculate convergence factor and time
    convs = np.zeros(len(allres))
    times = np.zeros(len(allres)+1)
    times[0] = tic
    for i in range(len(allres)):
        convs[i] = calc_conv(allres[i].r)
        times[i+1] = allres[i].t
    times = np.diff(times)
    for i in range(len(allres)):
        allres[i]._replace(t = times[i])
    labels = [ri.label for ri in allres]
    return convs, times, labels


def postprocess_residual_new(allres, tic):
    import numpy as np
    import pandas as pd

    #calculate convergence factor and time
    convs = np.zeros(len(allres))
    times = np.zeros(len(allres)+1)
    times[0] = tic
    for i in range(len(allres)):
        convs[i] = calc_conv(allres[i].r)
        times[i+1] = allres[i].t
    times = np.diff(times)
    for i in range(len(allres)):
        allres[i]._replace(t = times[i])
    labels = [ri.label for ri in allres]

    # put data into dataframe
    df = pd.DataFrame({"label":labels, "conv_fac":convs, "time":times, "residual": [ri.r for ri in allres]})

    return df


def postprocess_from_file(postfix,show_fig=True):
    import pandas as pd
    from pathlib import Path
    from script.utils.define_to_read_dir import to_read_dir, case_name
    from script.utils.plot_residuals import draw_times_new, plot_residuals_all_new
    dir = Path(to_read_dir).parent
    dir2 = dir/"png"
    file = dir2/Path("allres_"+postfix+".csv")
    df = pd.read_csv(file)

    df['residual'] = df['residual'].apply(lambda x: [float(y) for y in eval(x)])

    save_data_new(df,postfix)
    draw_times_new(df)
    
    plot_residuals_all_new(df, show_fig=show_fig,save_fig=True,postfix=postfix, use_markers=True)
    print_df_new(df)


def print_allres_time(allres, draw=False):
    import pandas as pd
    print("\n\nDataframe of time taken for each solver")
    pd.set_option("display.precision", 3)
    labels = [ri.label for ri in allres]
    times = [ri.t for ri in allres]
    df = pd.DataFrame({"label":labels, "time":times})
    print("\nIn increasing order of time taken:")
    df = df.sort_values(by="time", ascending=True)
    print(df)
    if draw:
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
    return df