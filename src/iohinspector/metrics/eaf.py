
from iohinspector.align import align_data
from iohinspector.metrics import transform_fval, get_sequence
import numpy as np
import pandas as pd


def get_discritized_eaf_single_objective(
        data,
        fval_var: str = "raw_y",
        eval_var: str = "evaluations",
        eval_values = None,
        eval_min = None,
        eval_max = None,
        eval_targets = 10,
        scale_eval_log: bool = True,
        f_min = 1e-8,
        f_max = 1e2,
        scale_f_log: bool = True,  
        f_targets = 101,
        ):

    if eval_values is None:
        if eval_min is None:
            eval_min = data[eval_var].min()
        if eval_max is None:
            eval_max = data[eval_var].max()
        eval_values = get_sequence(
            eval_min, eval_max, eval_targets, scale_log=scale_eval_log, cast_to_int=True
        )
    
    dt_aligned = align_data(
       data,
       eval_values,
       x_col=eval_var,
       y_col=fval_var,
       output="long"
       ) 
    dt_aligned = transform_fval(
        dt_aligned,
        lb=f_min,
        ub=f_max,
        scale_log=scale_f_log,
        fval_col=fval_var,
        )
    targets = np.linspace(0, 1, f_targets) 
    dt_targets = pd.DataFrame(targets, columns=["eaf_target"])

    dt_merged = dt_targets.merge(dt_aligned[[eval_var, 'eaf']].to_pandas(), how='cross')
    dt_merged['ps'] = dt_merged['eaf_target'] <= dt_merged['eaf']
    dt_discr = dt_merged.pivot_table(index='eaf_target', columns=eval_var, values='ps')

    return dt_discr
