import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from matplotlib.patches import Polygon
import seaborn as sbs
import matplotlib.pyplot as plt
font = {'size'   : 24}
plt.rc('font', **font)

from manager import DataManager
from typing import Iterable

import numpy as np

def plot_paretofronts_2d(data : DataManager,
                      hue_vars : Iterable[str],
                      style_var : str = None):
    assert (data.fval_vars) == 2

    required_cols = data.fval_vars + data.eval_vars + data.free_vars

    df = data.get_cumulative_nondominated()

    plt.figure(figsize=(16,9))
    sbs.scatterplot(df, x=data.fval_vars[0], y=data.fval_vars[1],
                    hue=hue_vars, style=style_var)
    plt.show()
    return df

def plot_eaf(data : DataManager,
             n_quantiles : int = 15):
    
    evals = get_sequence(1000, 1000000, 50, True, True)
    long = align_data(data.load(), np.array(evals, 'uint64'), ['data_id'], output='long')

    quantiles = np.arange(0,1+1/((n_quantiles-1)*2), 1/(n_quantiles-1))
    fig, ax = plt.subplots(figsize=(16,9))
    colors = sbs.color_palette('viridis', n_colors=len(quantiles))
    for quant, color in zip(quantiles, colors[::-1]):
        poly = np.array(long.group_by(data.eval_var).quantile(quant).sort(data.eval_var))
        poly = np.append(poly, np.array([[max(poly[:,0]), long[data.fval_var].max()]]), axis=0)
        poly = np.append(poly, np.array([[min(poly[:,0]), long[data.fval_var].max()]]), axis=0)
        poly2 = np.repeat(poly,2, axis=0)
        poly2[2::2, 1] = poly[:,1][:-1]
        ax.add_patch(Polygon(poly2, facecolor = color))
    plt.ylim(long[data.fval_var].min(),long[data.fval_var].max())
    plt.xlim(min(evals),max(evals))
    ax.set_axisbelow(True)
    plt.grid(which='both', zorder=100)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    return long