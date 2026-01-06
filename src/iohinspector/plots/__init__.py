import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
font = {"size": 24}
plt.rc("font", **font)


from .fixed_target import plot_single_function_fixed_target
from .fixed_budget import plot_single_function_fixed_budget
from .ecdf import plot_ecdf
from .eaf import plot_eaf_single_objective, plot_eaf_pareto, plot_eaf_diffs
from .multi_objective import plot_paretofronts_2d, plot_indicator_over_time
from .ranking import plot_tournament_ranking, plot_robustrank_over_time, plot_robustrank_changes
from .attractor_network import plot_attractor_network
from .single_run import plot_heatmap_single_run
from .utils import BasePlotArgs, LinePlotArgs