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
    long = _eaf(data, output='long')[[data.eval_variable, data.fval_variables[0]]]

    quantiles = np.arange(0,1+1/((n_quantiles-1)*2), 1/(n_quantiles-1))
    fig, ax = plt.subplots(figsize=(16,9))
    colors = sbs.color_palette('viridis', n_colors=len(quantiles))
    for quant, color in zip(quantiles[::-1], colors[::-1]):
        poly = np.array(long.group_by(data.eval_variable).quantile(quant).sort(data.eval_variable))
        poly = np.append(poly, np.array([[max(poly[:,0]), 0]]), axis=0)
        poly2 = np.repeat(poly,2, axis=0)
        poly2[2::2, 1] = poly[:,1][:-1]
        ax.add_patch(Polygon(poly2, facecolor = color))
    plt.ylim(0,1)
    plt.xlim(1,data.budget)
    plt.xscale('log')
    plt.show()
    return long