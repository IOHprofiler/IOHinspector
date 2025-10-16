from skelo.model.elo import EloEstimator
import numpy as np
import pandas as pd
import polars as pl
from typing import Iterable




def get_tournament_ratings(
    data: pl.DataFrame,
    alg_vars: Iterable[str] = ["algorithm_name"],
    fid_vars: Iterable[str] = ["function_name"],
    perf_var: str = "raw_y",
    nrounds: int = 25,
    maximization: bool = False,
):
    """Method to calculate ratings of a set of algorithm on a set of problems.
    Calculated based on nrounds of competition, where in each round all algorithms face all others (pairwise) on every function.
    For each round, a sampled performance value is taken from the data and used to determine the winner.
    This function uses the ELO rating scheme, as opposed to the Glicko2 scheme used in the IOHanalyzer. Deviations are estimated based on the last 5% of rounds.

    Args:
        data (pl.DataFrame): The data object to use for getting the performance.
        alg_vars (Iterable[str], optional): Which variables specific the algortihms which will compete. Defaults to ["algorithm_name"].
        fid_vars (Iterable[str], optional): Which variables denote the problems on which will be competed. Defaults to ["function_name"].
        perf_var (str, optional): Which variable corresponds to the performance. Defaults to "raw_y".
        nrounds (int, optional): How many round should be played. Defaults to 25.
        maximization (bool, optional): Whether the performance metric is being maximized. Defaults to False.

    Returns:
        pd.DataFrame: Pandas dataframe with rating, deviation and volatility for each 'alg_vars' combination
    """
    fids = data[fid_vars].unique()
    aligned_comps = data.pivot(
        index=alg_vars,
        columns=fid_vars,
        values=perf_var,
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
    return rating_dt_elo
