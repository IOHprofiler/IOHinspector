from iohinspector.indicators import add_indicator
from skelo.model.elo import EloEstimator
import numpy as np
import pandas as pd
import polars as pl
from typing import Iterable




def get_tournament_ratings(
    data: pl.DataFrame,
    alg_vars: Iterable[str] = ["algorithm_name"],
    fid_vars: Iterable[str] = ["function_name"],
    fval_var: str = "raw_y",
    nrounds: int = 25,
    maximization: bool = False,
    return_as_pandas: bool = True,
) -> pl.DataFrame | pd.DataFrame:
    """Calculate ELO tournament ratings for algorithms competing on multiple problems.

    Args:
        data (pl.DataFrame): The data object containing algorithm performance data.
        alg_vars (Iterable[str], optional): Which columns specify the algorithms that will compete. Defaults to ["algorithm_name"].
        fid_vars (Iterable[str], optional): Which columns denote the problems on which competition occurs. Defaults to ["function_name"].
        fval_var (str, optional): Which column contains the performance values. Defaults to "raw_y".
        nrounds (int, optional): Number of tournament rounds to play. Defaults to 25.
        maximization (bool, optional): Whether the performance metric is being maximized. Defaults to False.
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pd.DataFrame or pl.DataFrame: A DataFrame with ELO ratings, deviations, and algorithm identifiers.
    """
    fids = data[fid_vars].unique()
    aligned_comps = data.pivot(
        index=alg_vars,
        on=fid_vars,
        values=fval_var,
        aggregate_function=pl.element(),
    )
    players = aligned_comps[alg_vars]
    n_players = players.shape[0]
    comp_arr = np.array(aligned_comps[aligned_comps.columns[len(alg_vars) :]])

    rng = np.random.default_rng()
    fids = [i for i in range(len(fids))]
    lplayers = [i for i in range(n_players)]
    records = []
    for r in range(nrounds):
        for fid in fids:
            for p1 in lplayers:
                for p2 in lplayers:
                    if p1 == p2:
                        continue
                    s1 = rng.choice(comp_arr[p1][fid], 1)[0]
                    s2 = rng.choice(comp_arr[p2][fid], 1)[0]
                    if s1 == s2:
                        won = 0.5
                    else:
                        won = abs(float(maximization) - float(s1 < s2))

                    records.append([r, p1, p2, won])
    dt_comp = pd.DataFrame.from_records(
        records, columns=["round", "p1", "p2", "outcome"]
    )
    dt_comp = dt_comp.sample(frac=1).sort_values("round")
    model = EloEstimator(key1_field="p1", key2_field="p2", timestamp_field="round").fit(
        dt_comp, dt_comp["outcome"]
    )
    model_dt = model.rating_model.to_frame()
    ratings = np.array(model_dt[np.isnan(model_dt["valid_to"])]["rating"])
    deviations = (
        model_dt.query(f"valid_from >= {nrounds * 0.95}").groupby("key")["rating"].std()
    )

    rating_dt_elo = pd.DataFrame(
        [
            ratings,
            deviations,
            *players[players.columns],
        ]
    ).transpose()
    rating_dt_elo.columns = ["Rating", "Deviation", *players.columns]
    if return_as_pandas:
        return rating_dt_elo
    else:
        rating_dt_elo_pl = pl.from_pandas(rating_dt_elo)
        return rating_dt_elo_pl


def get_robustrank_over_time(
        data: pl.DataFrame,
        obj_vars: Iterable[str],
        evals: Iterable[int],
        indicator: object,
    
):
    """Calculate robust ranking data over multiple time points for multi-objective optimization.

    Args:
        data (pl.DataFrame): The data object containing multi-objective optimization trajectory data.
        obj_vars (Iterable[str]): Which columns correspond to the objective values.
        evals (Iterable[int]): Evaluation time points at which to calculate rankings.
        indicator (object): Indicator object from iohinspector.indicators for performance measurement.

    Returns:
        tuple: A tuple containing (comparison, benchmark) objects for robust ranking analysis.
    """
    from robustranking import Benchmark
    from robustranking.comparison import MOBootstrapComparison

    df = add_indicator(
        data, indicator, obj_vars=obj_vars, evals=evals
    ).to_pandas()
    df_part = df[["evaluations", indicator.var_name, "algorithm_name", "run_id"]]
    dt_pivoted = pd.pivot(
        df_part,
        index=["algorithm_name", "run_id"],
        columns=["evaluations"],
        values=[indicator.var_name],
    ).reset_index()
    dt_pivoted.columns = ["algorithm_name", "run_id"] + evals
    benchmark = Benchmark()
    benchmark.from_pandas(dt_pivoted, "algorithm_name", "run_id", evals)
    comparison = MOBootstrapComparison(
        benchmark,
        alpha=0.05,
        minimise=indicator.minimize,
        bootstrap_runs=1000,
        aggregation_method=np.mean,
    )
    
    return comparison,  benchmark


def get_robustrank_changes(  
    data: pl.DataFrame,
    obj_vars: Iterable[str],
    evals: Iterable[int],
    indicator: object,
    ):
    """Calculate robust ranking changes across multiple evaluation time points.

    Args:
        data (pl.DataFrame): The data object containing multi-objective optimization trajectory data.
        obj_vars (Iterable[str]): Which columns correspond to the objective values.
        evals (Iterable[int]): Evaluation time points at which to calculate ranking changes.
        indicator (object): Indicator object from iohinspector.indicators for performance measurement.

    Returns:
        dict: A dictionary of comparison objects for each evaluation time point showing ranking changes.
    """
    from robustranking import Benchmark
    from robustranking.comparison import BootstrapComparison

    df = add_indicator(
        data, indicator, obj_vars=obj_vars, evals=evals
    ).to_pandas()
    df_part = df[["evaluations", indicator.var_name, "algorithm_name", "run_id"]]
    dt_pivoted = pd.pivot(
        df_part,
        index=["algorithm_name", "run_id"],
        columns=["evaluations"],
        values=[indicator.var_name],
    ).reset_index()
    dt_pivoted.columns = ["algorithm_name", "run_id"] + evals

    comparisons = {
        f"{eval}": BootstrapComparison(
            Benchmark().from_pandas(dt_pivoted, "algorithm_name", "run_id", eval),
            alpha=0.05,
            minimise=indicator.minimize,
            bootstrap_runs=1000,
        )
        for eval in evals
    }

    return comparisons