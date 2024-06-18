def _convergence(data : DataSet, 
                    measure : str,
                     ):
    x_values = get_sequence(data.min_budget, data.max_budget, 50, 'budget')
    data_aligned = data.get_aligned(x_values, output='wide')
    dt_plot = wide.groupby(['ID']).agg(['mean', 'min', 'max', 'median', 'std'], axis='columns')

    plt.figure(figsize=(16,9))
    sbs.lineplot(dt_plot, x='evaluations', y=measure, hue='ID')
    plt.show()
    return dt_plot

def plot_eaf(data : DataSet, 
                     ):
    x_values = get_sequence(data.min_budget, data.max_budget, 50, 'budget')
    data_aligned = data.get_aligned(x_values, output='wide')
    dt_plot = eaf_transform(wide).groupby(['ID']).agg(['mean'], axis='columns')

    return dt_plot


def get_aocc(data : DataSet, 
                     ):
    dt_plot = data.get_aocc()

def calc_auc(group):
    new_row = pd.DataFrame({'evaluations' : max_budget, 'raw_y' : min(group['raw_y'])}, index=[0])
    group = pd.concat([group, new_row], ignore_index=True)
    group = group.query(f"evaluations <= {max_budget}")
    auc = pd.DataFrame({'auc' : [np.sum((upper_bound - np.clip(np.log10(group['raw_y'][:-1]), -8, upper_bound)) * np.ediff1d(group['evaluations']))/(max_budget*(8+upper_bound)) ] })
    # print(group, auc)
    for col in ['run_id']:
        auc[col] = group[col].iloc[0]
    return auc



def group_hv(group):
    group = group.with_columns(
        pl.Series(name="igd", values=[HV([100,100])(np.array(group[['y1', 'y2']])[:i]) for i in range(1, len(group)+1)])
    )
    return group
    
group_hv(df2.filter(pl.col('run_id') == 1))