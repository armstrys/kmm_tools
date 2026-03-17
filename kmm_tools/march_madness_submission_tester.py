# %% [code]
"""
This utility script provides some useful utilities for benchmarking submissions.

You can find notebook examples of usage here:
https://www.kaggle.com/code/rsa013/march-madness-submission-benchmark-example/edit/run/222853167
"""

from typing import Sequence, Literal
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import functools
from itertools import combinations
import os


######### sets competition path automatically #########
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass
if (p := os.getenv("KAGGLE_KERNEL_RUN_TYPE")) is not None:
    DEFAULT_COMPETITION_DATA_PATH = Path(
        "/kaggle/input/march-machine-learning-mania-2025"
    )
    # COMPETITION_DATA_PATH = Path("/kaggle/input/march-machine-learning-mania-2025")
elif (p := os.getenv("COMPETITION_DATA_PATH")) is not None:
    DEFAULT_COMPETITION_DATA_PATH = Path(p)
else:
    raise RuntimeError(
        "If running locally you must define an environment variable COMPETITION_DATA_PATH."
    )
#######################################################


def evaluate_stage1_submission(
    submission: pd.DataFrame,
    seasons: Sequence[int] = (2021, 2022, 2023, 2024),
    mode: Literal["brier", "logloss"] = "brier",
) -> float:
    """
    Evaluate a Stage 1 submission for a forecasting competition.

    Parameters
    ----------
    submission : pd.DataFrame
        The submission DataFrame, containing forecasted probabilities.
    seasons : Sequence[int], optional
        The seasons to evaluate the submission for. Defaults to (2021, 2022, 2023, 2024).
    mode : Literal["brier", "logloss"], optional
        The evaluation metric to use. "brier" for Brier score, "logloss" for log loss. Defaults to "brier".

    Returns
    -------
    score : float
        The resulting score from the historical tournament results, predictions, and chosen parameters.
    """
    scores = evaluate_stage1_submission_games(
        submission=submission,
        seasons=seasons,
        mode=mode,
    )
    return np.array(list(scores.values())).mean()


def evaluate_stage1_submission_games(
    submission: pd.DataFrame,
    seasons: Sequence[int] = (2021, 2022, 2023, 2024),
    mode: Literal["brier", "logloss"] = "brier",
) -> dict[str, float]:
    """
    Evaluate scores for each game in a submission.

    Parameters
    ----------
    submission : pd.DataFrame
        The submission DataFrame, containing forecasted probabilities.
    seasons : Sequence[int], optional
        The seasons to evaluate the submission for. Defaults to (2021, 2022, 2023, 2024).
    mode : Literal["brier", "logloss"], optional
        The evaluation metric to use. "brier" for Brier score, "logloss" for log loss. Defaults to "brier".

    Returns
    -------
    scores : dict[str, float]
        A dictionary with the resulting score from the historical tournament results, predictions, and chosen parameters
        for each game.
    """
    prediction_dict = submission.set_index("ID")["Pred"].to_dict()
    results_dict = _get_historical_results(seasons=seasons)
    match mode:
        case "brier":
            f = _brier
        case "logloss":
            f = _logloss
    return {gid: f(prediction_dict[gid], r) for gid, r in results_dict.items()}


def validate_submission_format(
    submission: pd.DataFrame, check_seasons: Sequence[int] = [2025]
) -> dict[str, bool]:
    """
    Validate that a submission is ready for submission.

    This function checks the format of the submission to ensure it meets the requirements of the competition.
    Use this to verify your submission before uploading it to Kaggle.

    Note that this is not endorsed by Kaggle and the function may have errors. Please reach out if you suspect
    there are any issues.

    Parameters
    ----------
    submission : pd.DataFrame
        The submission DataFrame to validate.
    check_seasons : Sequence[int], optional
        The seasons to validate the submission for. Defaults to [2025].

    Returns
    -------
    validation_results : dict[str, bool]
        A dictionary with the validation results.
    """
    checks = {
        "validation info": {
            "checking seasons": check_seasons,
            "columns": submission.columns.tolist(),
            "row count": len(submission),
        },
    }
    checks["correct columns"] = _check_columns(submission)
    checks["correct team order in ids"] = _check_id_team_order(submission)
    checks["correct row count"] = _check_sub_count(submission, check_seasons)
    return checks


def make_template_submission(seasons: Sequence[int]) -> pd.DataFrame:
    """
    Creates a template submission DataFrame for the specified seasons.

    Parameters
    ----------
    seasons : Sequence[int]
        The seasons for which to create a template submission.

    Returns
    -------
    template_submission : pd.DataFrame
        A template submission DataFrame with the correct columns and indices.
    """
    game_ids = []
    for s in seasons:
        m_teams, w_teams = _get_season_teams(s)
        for teams in (m_teams, w_teams):
            game_ids.extend(f"{s}_{a}_{b}" for a, b in combinations(teams, 2))
    sub = pd.DataFrame({"ID": game_ids, "Pred": np.full(len(game_ids), np.nan)})
    return sub


#########################################################
#      Private functions and tests below this point


@functools.cache
def _cached_csv_read(path: Path) -> pd.DataFrame:
    """cached read to improve performance"""
    return pd.read_csv(path)


def _get_historical_results(
    seasons: Sequence[int] = (2021, 2022, 2023, 2024),
) -> dict[str, int]:
    """
    parse a list of historical results to the expected kaggle result format in a dict
    """
    tourney_df = pd.concat(
        [
            _cached_csv_read(
                DEFAULT_COMPETITION_DATA_PATH / f"{mw}NCAATourneyDetailedResults.csv"
            )
            for mw in "MW"
        ]
    )
    seeds_df = pd.concat(
        [
            _cached_csv_read(
                DEFAULT_COMPETITION_DATA_PATH / f"{mw}NCAATourneySeeds.csv"
            )
            for mw in "MW"
        ]
    )
    # filter unneeded rows/columns
    tourney_df = tourney_df.loc[
        tourney_df["Season"].isin(seasons), ["Season", "WTeamID", "LTeamID"]
    ].copy()
    # add seeds to find play in games
    for wl in "WL":
        tourney_df = pd.merge(
            left=tourney_df,
            right=seeds_df.add_suffix(f"_{wl}")[
                [f"Season_{wl}", f"TeamID_{wl}", f"Seed_{wl}"]
            ],
            how="left",
            left_on=["Season", f"{wl}TeamID"],
            right_on=[f"Season_{wl}", f"TeamID_{wl}"],
        ).copy()
    # filter play-in games out of results
    play_in_game_mask = tourney_df["Seed_W"].map(
        lambda x: x in ("a", "b")
    ) & tourney_df["Seed_L"].map(lambda x: x in ("a", "b"))
    kaggle_results = tourney_df[~play_in_game_mask].copy()
    # get the game ids for all games - all results are 1
    w_ids = (
        kaggle_results["Season"].astype(str)
        + "_"
        + kaggle_results["WTeamID"].astype(str)
        + "_"
        + kaggle_results["LTeamID"].astype(str)
    )
    w_ids.to_list()
    # flip game ids and results so that lowest team id is listed first
    dont_flip = (tourney_df["WTeamID"] < tourney_df["LTeamID"]).to_list()
    return {wid if df else _swap_game_id(wid): df for wid, df in zip(w_ids, dont_flip)}


def _swap_game_id(game_id: str) -> str:
    """swaps team a and b in a game id"""
    parts = game_id.split("_")
    parts.append(parts.pop(1))
    return "_".join(parts)


def _brier(pred: float, result: int) -> float:
    """calculates brier score"""
    return (pred - result) ** 2


def _logloss(pred: float, result: int) -> float:
    """calculates log loss"""
    return (result * -np.log(pred)) + ((1 - result) * -np.log(1 - pred))


def _check_columns(submission: pd.DataFrame) -> bool:
    """checks that columns are ID and Pred"""
    return set(submission.columns) == {"ID", "Pred"}


def _check_id_team_order(submission: pd.DataFrame) -> bool:
    """checks that team A ID is greater than team B ID"""
    return (
        submission["ID"]
        .map(lambda gid: int((parts := gid.split("_"))[1]) < int(parts[2]))
        .any()
    )


def _check_sub_count(
    submission: pd.DataFrame,
    seasons: Sequence[int] = (2021, 2022, 2023, 2024),
) -> bool:
    """checks the number of rows in a submission"""
    team_counts = [_get_season_teams(s) for s in seasons]
    possible_match_counts = [
        _comb_count(len(m)) + _comb_count(len(w)) for m, w in team_counts
    ]
    expected_count = sum(possible_match_counts)
    return len(submission) == expected_count


def _comb_count(n: int) -> int:
    """calculates the number of unique pairs"""
    return int((n * (n - 1)) / 2)


def _get_season_teams(season: int) -> tuple[list[int], list[int]]:
    """gets the unique teams for both men and women in a given season"""
    m_conference_teams = _cached_csv_read(
        DEFAULT_COMPETITION_DATA_PATH / "MTeamConferences.csv"
    )
    w_conference_teams = _cached_csv_read(
        DEFAULT_COMPETITION_DATA_PATH / "WTeamConferences.csv"
    )
    return (
        m_conference_teams[m_conference_teams["Season"] == season]["TeamID"]
        .astype(int)
        .to_list(),
        w_conference_teams[w_conference_teams["Season"] == season]["TeamID"]
        .astype(int)
        .to_list(),
    )


#### tests ####


def _test_sample_evaluation() -> None:
    """make sure sample submissions return expected values"""
    expectations = {
        DEFAULT_COMPETITION_DATA_PATH / "SampleSubmissionStage1.csv": 0.25,
    }
    for sample_path, expected_result in expectations.items():
        if not sample_path.exists():
            warnings.warn("{sample_path} does not exist... skipping tests")
        sample = _cached_csv_read(sample_path)
        seasons = sample["ID"].map(lambda x: int(x.split("_")[0])).unique().tolist()
        score = evaluate_stage1_submission(sample, seasons)
        assert np.isclose(score, expected_result), (
            f"tests not returning expected results: {score} != {expected_result}"
        )


_test_sample_evaluation()


def _test_submission_count() -> None:
    """make sure the row count checker agrees with provided samples"""
    paths = (
        DEFAULT_COMPETITION_DATA_PATH / "SampleSubmissionStage1.csv",
        DEFAULT_COMPETITION_DATA_PATH / "SampleSubmissionStage2.csv",
    )
    for sample_path in paths:
        if not sample_path.exists():
            warnings.warn("{sample_path} does not exist... skipping tests")
        sample = _cached_csv_read(sample_path)
        seasons = sample["ID"].map(lambda x: int(x.split("_")[0])).unique().tolist()
        assert not _check_sub_count(sample.head(), seasons), (
            "something is wrong with count checker"
        )
        assert _check_sub_count(sample, seasons), (
            "something is wrong with count checker"
        )


_test_submission_count()


def _test_submission_team_order() -> None:
    """make sure order checker agrees with provided samples"""
    paths = (
        DEFAULT_COMPETITION_DATA_PATH / "SampleSubmissionStage1.csv",
        DEFAULT_COMPETITION_DATA_PATH / "SampleSubmissionStage2.csv",
    )
    for sample_path in paths:
        if not sample_path.exists():
            warnings.warn("{sample_path} does not exist... skipping tests")
        sample = _cached_csv_read(sample_path)
        bad_sample = sample.copy()
        bad_sample["ID"] = bad_sample["ID"].map(_swap_game_id)

        assert _check_id_team_order(sample.head()), (
            "something is wrong with team order checker"
        )
        assert not _check_id_team_order(bad_sample.head()), (
            "something is wrong with team order checker"
        )


_test_submission_team_order()


def _test_submission_columns() -> None:
    """make sure column checker agrees with provided samples"""
    paths = (
        DEFAULT_COMPETITION_DATA_PATH / "SampleSubmissionStage1.csv",
        DEFAULT_COMPETITION_DATA_PATH / "SampleSubmissionStage2.csv",
    )
    for sample_path in paths:
        if not sample_path.exists():
            warnings.warn("{sample_path} does not exist... skipping tests")
        sample = _cached_csv_read(sample_path)
        assert _check_columns(sample.head()), (
            "something is wrong with team order checker"
        )
        bad_sample = sample.copy()
        assert not _check_columns(bad_sample.rename(columns={"Pred": "P"}).head()), (
            "something is wrong with team order checker"
        )
        bad_sample["extra_column"] = True
        assert not _check_columns(bad_sample.head()), (
            "something is wrong with team order checker"
        )


_test_submission_columns()


def test_submission_creation() -> None:
    """creates a sample template and ensures it is consistent with checks"""
    stage1_seasons = (2021, 2022, 2023, 2024, 2025)
    stage1_sub = make_template_submission(stage1_seasons)
    assert _check_columns(stage1_sub)
    assert _check_id_team_order(stage1_sub)
    assert _check_sub_count(stage1_sub, stage1_seasons)


test_submission_creation()
