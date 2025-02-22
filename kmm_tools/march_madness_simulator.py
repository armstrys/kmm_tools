"""
Classes and functions to easily and quickly
simulate NCAA tournament games using kaggle data
"""

import dataclasses
import pandas as pd
from pathlib import Path
from typing import Literal, Sequence
import re
import numpy as np
import graphviz
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

ROUND_NAMES = {
    0: "Play-in Games",
    1: "First Round",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final 4",
    6: "Championship",
}


@dataclasses.dataclass(frozen=True)
class Team:
    """A dataclass representing a team."""

    id: int
    name: str
    seed: str

    def __repr__(self):
        return f"{self.seed} {self.name} - TeamID: {self.id}"


@dataclasses.dataclass()
class Game:
    """A dataclass representing a game."""

    season: int
    slot: str
    strong_seed: str
    weak_seed: str
    strong_team: Team | None = None
    weak_team: Team | None = None
    winner: Team | None = None
    outcome_probability: float | None = None

    @property
    def game_id(self):
        return "_".join(
            [str(self.season), str(self.strong_team.id), str(self.weak_team.id)]
        )

    @property
    def r(self):
        """extract round label from slot"""
        r = re.compile(r"R(.)[WXYZC].")
        match = r.search(self.slot)
        r = int(match.group(1)) if match is not None else 0
        return r


@dataclasses.dataclass
class Tournament:
    games: list[Game]
    probabilities: dict[str, float]  # gameid: probability

    @property
    def results(self):
        return {g.slot: g.winner for g in self.games}


######## tournament structure #######


def start_tournament(
    submission: pd.DataFrame,
    season: int,
    mw=Literal["M", "W"],
    competition_data_path: Path = DEFAULT_COMPETITION_DATA_PATH,
) -> Tournament:
    """
    Start a tournament simulation.

    Args:
        submission (pd.DataFrame): The submission dataframe.
        season (int): The season number.
        mw (Literal["M", "W"]): The men's or women's tournament.
        competition_data_path (Path, optional): The path to the competition data.
            Defaults to DEFAULT_COMPETITION_DATA_PATH.

    Returns:
        Tournament: The simulated tournament.
    """
    slots, teams, seeds = _get_tournament_data(
        season=season, mw=mw, competition_data_path=competition_data_path
    )
    seed_team_dict = seeds.set_index("Seed")["TeamID"].to_dict()
    team_dict = teams.set_index("TeamID")["TeamName"].to_dict()
    games = _make_tournament_games(slots=slots, teams=teams, seeds=seeds)
    tournament = Tournament(
        games=games,
        probabilities=submission.set_index("ID")["Pred"].to_dict(),
    )
    for g in games:
        strong_id = seed_team_dict.get(g.strong_seed)
        weak_id = seed_team_dict.get(g.weak_seed)
        if strong_id is not None:
            g.strong_team = Team(
                id=strong_id,
                name=team_dict[strong_id],
                seed=g.strong_seed,
            )
        if weak_id is not None:
            g.weak_team = Team(
                id=weak_id,
                name=team_dict[weak_id],
                seed=g.weak_seed,
            )
    return tournament


def _get_tournament_data(
    season: int,
    mw=Literal["M", "W"],
    competition_data_path: Path = DEFAULT_COMPETITION_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieves tournament data for a given competition, season, and men's/women's bracket.

    Args:
    competition_path (Path): The path to the competition data.
    season (int): The season year.
    mw (Literal["M", "W"]): The men's or women's bracket.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple of three dataframes containing tournament data.
    """
    if not isinstance(competition_data_path, Path):
        competition_data_path = Path(competition_data_path)
    seeds = pd.read_csv(competition_data_path / (f"{mw}NCAATourneySeeds.csv")).query(
        "Season == @season"
    )
    tournament_teams = seeds["TeamID"].unique().tolist()
    teams = pd.read_csv(competition_data_path / (f"{mw}Teams.csv")).query(
        "TeamID == @tournament_teams"
    )
    slots = pd.read_csv(competition_data_path / (f"{mw}NCAATourneySlots.csv")).query(
        "Season == @season"
    )
    return slots, teams, seeds


def _make_tournament_games(
    slots: pd.DataFrame, teams: pd.DataFrame, seeds: pd.DataFrame
) -> list[Game]:
    return [
        Game(
            season=r["Season"],
            slot=r["Slot"],
            strong_seed=r["StrongSeed"],
            weak_seed=r["WeakSeed"],
        )
        for _, r in slots.iterrows()
    ]


def _swap_game_id(game_id: str) -> str:
    """swaps team a and b in a game id"""
    parts = game_id.split("_")
    parts.append(parts.pop(1))
    return "_".join(parts)


####### game simulation #######


def simulate_game(
    game: Game,
    probabilities: dict[str, float],
    style: Literal["random", "chalk", "results"],
    results: dict[str, int] = None,
) -> None:
    """
    Simulates a game based on the given style and probabilities.

    Args:
        game (Game): The game to simulate.
        probabilities (dict[str, float]): The probabilities for each game.
        style (Literal["random", "chalk", "results"]): The style of simulation.
        results (dict[str, int]): The results of the tournament.

    Returns:
        Team: The winning team.
    """
    p_strong = _get_game_probability(game, probabilities)
    match style:
        case "chalk":
            if p_strong >= 0.5:
                game.winner = game.strong_team
                game.outcome_probability = p_strong
            else:
                game.winner = game.weak_team
                game.outcome_probability = 1 - p_strong

        case "random":
            if p_strong >= np.random.rand():
                game.winner = game.strong_team
                game.outcome_probability = p_strong
            else:
                game.winner = game.weak_team
                game.outcome_probability = 1 - p_strong
        case "results":
            if results is None:
                raise ValueError("to use this style (results) you must provide results")
            w_id = results[game.slot]
            if w_id == game.strong_team.id:
                game.winner = game.strong_team
            elif w_id == game.weak_team.id:
                game.winner = game.weak_team
            else:
                raise ValueError(
                    f"result team id {w_id} is not a valid option for {game}"
                )
        case _:
            raise ValueError(
                f"{style=} not recognized. " 'Should be "chalk" or "random"'
            )


def switch_winner(game: Game) -> None:
    """switches the winner of a game"""
    game.winner = (
        game.weak_team if game.winner.id == game.strong_team.id else game.strong_team
    )
    game.outcome_probability = 1 - game.outcome_probability


def _get_game_probability(game: Game, probabilities: dict[str, float]) -> float:
    p_strong = probabilities.get(game.game_id)
    if p_strong is None:
        p_strong = 1 - probabilities.get(_swap_game_id(game.game_id))
        if p_strong is None:
            raise RuntimeError(f"Game ID {game.game_id} not found in submission")
    return p_strong


####### multiple games ######


def simulate_round(
    tournament: Tournament,
    r: int,
    style: Literal["random", "chalk", "results"],
    results: dict[str, int] | None = None,
) -> None:
    """
    Simulates a round in the tournament based on the given style.

    Args:
        tournament (Tournament): The tournament to simulate.
        r (int): The round to simulate.
        style (Literal["random", "chalk", "results"]): The style of simulation.
        results (dict[str, int]): The results of the tournament.

    Returns:
        Tournament: The updated tournament.
    """
    # reset games in later rounds
    for g in tournament.games:
        if g.r > r:
            g.winner = g.outcome_probability = None
        if g.r > (r + 1):
            g.strong_team = g.weak_team = None
    # simulate
    for g in tournament.games:
        if g.r == r:
            simulate_game(
                game=g,
                probabilities=tournament.probabilities,
                style=style,
                results=results,
            )
        _set_next_games(game=g, games=tournament.games)


def _set_next_games(game: Game, games: list[Game]) -> None:
    """
    Set the next games in the tournament based on the given game.

    Args:
        game (Game): The game to set the next games for.
        games (list[Game]): The list of games in the tournament.

    Returns:
        list[Game]: The updated list of games.
    """
    for set_g in games:
        if set_g.strong_seed == game.slot:
            set_g.strong_team = game.winner
        if set_g.weak_seed == game.slot:
            set_g.weak_team = game.winner


####### n simulate tournaments ########


def simulate_tournament(
    tournament: Tournament, style: Literal["random", "chalk", "results"]
) -> Tournament:
    """
    Simulates a tournament based on the given style.

    Args:
        tournament (Tournament): The tournament to simulate.
        style (Literal["random", "chalk", "results"]): The style of simulation.

    Returns:
        Tournament: The simulated tournament.
    """
    for r in range(7):
        simulate_round(tournament=tournament, r=r, style=style)
    return tournament


def simulate_n_tournaments(tournament: Tournament, n: int) -> pd.DataFrame:
    """
    Simulates multiple tournaments and returns a dataframe with the results.

    Args:
        tournament (Tournament): The tournament to simulate.
        n (int): The number of simulations to perform.

    Returns:
        pd.DataFrame: A dataframe with the summarized results of the simulated tournaments.
    """
    summary = {}
    for i in range(n):
        simulate_tournament(tournament=tournament, style="random")
        summary = summarize_results(
            results=tournament.results, previous_summary=summary
        )
        # [g for g in tournament.games if g.winner is not None]
    return summary_to_df(tournament, summary, n)


####### summary #######


def summarize_results(
    results: dict[str, int], previous_summary: dict[str, dict[int, int]] | None = None
) -> dict[str, dict[int, int]]:
    """
    Summarize the results of the tournament.

    Args:
        results (dict[str, int]): The results of the tournament.
        previous_summary (dict[str, dict[int, int]], optional): The previous summary. Defaults to None.

    Returns:
        dict[str, dict[int, int]]: The summarized results.
    """
    if previous_summary is not None:
        summary = previous_summary
    else:
        summary = {}
    for slot, team in results.items():
        team = team.id
        r = slot[:2]
        if "R" not in r:
            r = "R0"
        if summary.get(r) is None:
            summary.update({r: {team: 1}})
        elif summary[r].get(team) is None:
            summary[r].update({team: 1})
        else:
            summary[r][team] += 1
    return summary


def summary_to_df(
    tournament: Tournament, summary: dict[str, dict[int, int]], n_sim: int
) -> pd.DataFrame:
    """
    Convert the tournament summary to a pandas DataFrame.

    Args:
        tournament (Tournament): The tournament to summarize.
        summary (dict[str, dict[int, int]]): The summarized results.
        n_sim (int): The number of simulations.

    Returns:
        pd.DataFrame: The summarized results as a DataFrame.
    """
    columns = [ROUND_NAMES.get(k) for k in range(7)]
    if "R0" not in summary:
        columns = columns[1:]
    summary_df = pd.DataFrame(summary, dtype=np.float64)
    summary_df = summary_df[sorted(summary_df.columns)]
    summary_df.columns = columns
    summary_df.index.name = "TeamID"
    all_teams = list(
        {g.strong_team.id for g in tournament.games}.union(
            {g.weak_team.id for g in tournament.games}
        )
    )
    missing_teams = list(set(all_teams) - set(summary_df.index))
    if len(missing_teams) > 0:
        missing_teams_df = pd.DataFrame(
            np.nan, index=missing_teams, columns=summary_df.columns
        )
        summary_df = pd.concat([summary_df, missing_teams_df])
    summary_df["First Round"] = summary_df["First Round"].fillna(n_sim)
    summary_df = summary_df.fillna(0.0)
    summary_df.sort_values(by=columns[::-1], ascending=False, inplace=True)
    team_dict = {
        g.strong_team.id: g.strong_team.name
        for g in tournament.games
        if g.strong_team is not None
    }
    team_dict.update(
        {
            g.weak_team.id: g.weak_team.name
            for g in tournament.games
            if g.weak_team is not None
        }
    )
    summary_df.index = summary_df.index.map(lambda x: f"{team_dict.get(x)} ({x})")
    summary_df.iloc[:] /= n_sim
    return summary_df


def graph_games(tournament: Tournament, rounds: Sequence[int] = tuple(range(7))):
    """
    Visualize the tournament games.

    Args:
        tournament (Tournament): The tournament to visualize.
        rounds (Sequence[int], optional): The rounds to visualize. Defaults to tuple(range(7)).
    """
    games = [g for g in tournament.games if g.r in rounds]
    graph = graphviz.Digraph(node_attr={"shape": "rounded", "color": "lightblue2"})
    for g in games:
        t1 = "R" + f"{g.r} {g.strong_team.seed}-{g.strong_team.name}"
        t2 = "R" + f"{g.r} {g.weak_team.seed}-{g.weak_team.name}"
        w = "R" + f"{g.r+1} {g.winner.seed}" f"-{g.winner.name}"
        winner_params = {"color": "green", "label": f"{g.outcome_probability:.1%}"}
        loser_params = {"color": "red"}
        if g.strong_team.id == g.winner.id:
            t1_params = winner_params
            t2_params = loser_params
        else:
            t2_params = winner_params
            t1_params = loser_params
        graph.edge(t1, w, **t1_params)
        graph.edge(t2, w, **t2_params)
    graph.graph_attr["rankdir"] = "LR"
    graph.graph_attr["size"] = "30"
    graph.node_attr.update(style="rounded")
    return graph


# ##### loss ######


# ##### functional tests #######

# submission = pd.read_csv(
#     "/kaggle/input/march-machine-learning-mania-2025/SeedBenchmarkStage1.csv"
# )

# tournament = start_tournament(season=2023, mw="M", submission=submission)

# summary = simulate_n_tournaments(tournament, 100)
# print(summary.head(10))
# graph_games(tournament)
