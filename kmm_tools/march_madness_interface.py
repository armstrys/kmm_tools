"""
This is designed to run on a Kaggle notebook environment and will not work
locally without some edits. Use bracket_builder_streamlit.py
for a local installation
"""

# Data manipulation
import os
from pathlib import Path
import pandas as pd
import numpy as np
import base64
from march_madness_submission_tester import _brier, evaluate_stage1_submission
from march_madness_slot_results import HISTORIC_RESULTS
from march_madness_simulator import (
    start_tournament,
    simulate_game,
    _set_next_games,
    switch_winner,
    graph_games,
)

import streamlit as st

st.set_page_config(layout="wide")

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

# Info
"""
## NCAA Bracket Builder
Please look at the settings in the side bar to get started. You will add your
solution file there.

This GUI will walk you through the bracket game by game. The tabs at the top 
display the games by round. All games will be simulated using the settings in 
the side bar. If you want to change your selection for a game, click the button 
for the team of your choice.

The summary page will show some statistics about your bracket and let you 
display a visual representation. You can also download your results here.

If you want to run this locally you can see the 
[Github page](https://github.com/armstrys/kmm_tools) for more info.
"""

# Collect data
sub_file = st.sidebar.file_uploader(label="Drag your Kaggle solution file here")

# Prep data   ----- YOU DONT NEED TO CHANGE THIS
try:
    submission = pd.read_csv(sub_file)
except ValueError:
    st.warning("""
               Please add a valid solution file and
               adjust settings to continue!
               """)
    quit()


# Collect simulation info
seasons = submission["ID"].map(lambda x: int(x.split("_")[0])).unique().tolist()
season = int(st.sidebar.selectbox(label="Season", options=np.sort(seasons)[::-1]))
mw = (
    st.sidebar.segmented_control(
        label="Men's or Women's bracket?", options=["Men", "Women"], default="Men"
    )
)[0][0]


current_r = 0
tournament = start_tournament(
    submission=submission,
    season=season,
    mw=mw,
    competition_data_path=DEFAULT_COMPETITION_DATA_PATH,
)
try:
    historic_results = HISTORIC_RESULTS[mw][season]
except KeyError:
    historic_results = {}


with st.sidebar:
    st.divider()
    use_historic_results = False
    if len(historic_results) > 0:
        use_historic_results = st.checkbox("Apply historic results", value=False)

    style = st.segmented_control(
        "Stochastic or Deterministic Bracket?", ["Chalk", "Random"], default="Chalk"
    ).lower()
    seed = st.number_input(label="Seed for stochastic bracket:", value=0, min_value=0)
    np.random.seed(seed)

    st.write("""
                    **Chalk bracket**: will always select the team favored by the
                    model.\n
                    **Random bracket**: will randomize the winner for each game
                    using the model probabilities. Please choose a new seed to
                    change the randomization!
                    """)

if not np.isnan(sub_score := evaluate_stage1_submission(submission, seasons=seasons)):
    st.markdown(f"### Total submission score: {sub_score:.5f}")

SEASON_INFO = pd.read_csv(DEFAULT_COMPETITION_DATA_PATH / (f"{mw}Seasons.csv")).query(
    "Season == @season"
)
REGION_DICT = {
    "W": "W " + SEASON_INFO["RegionW"].values[0],
    "X": "X " + SEASON_INFO["RegionX"].values[0],
    "Y": "Y " + SEASON_INFO["RegionY"].values[0],
    "Z": "Z " + SEASON_INFO["RegionZ"].values[0],
}
# Initialize simulation
sim_headers = {
    0: "First Four",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Finals",
}


t1, t2, t3, t4, t5, t6, ts = st.tabs(
    list(sim_headers.values())[:5] + ["Final Four and Championship", "Summary"]
)
round_tabs = {
    0: t1,
    1: t2,
    2: t3,
    3: t4,
    4: t5,
    5: t6,
    6: t6,
}


if len(tournament.games) == 63 and current_r == 0:
    current_r = 1
    t1.markdown("#### No Games - Proceed to next round.")
# Run simulation
while current_r < 7:
    t = round_tabs[current_r]
    t.header(sim_headers[current_r])
    for g in tournament.games:
        if g.r == current_r:
            simulate_game(g, tournament.probabilities, style=style)
            if use_historic_results and g.slot in historic_results.keys():
                historic_winner_id = historic_results[g.slot]
                if g.winner.id != historic_winner_id:
                    switch_winner(g)
            s = g.strong_team
            w = g.weak_team
            t.markdown(
                f"**{g.slot}** - {s.seed} **{s.name}** vs. {w.seed} **{w.name}**"
            )
            override = t.segmented_control(
                label="Winner:",
                options=[s.name, w.name],
                default=g.winner.name,
            )
            if override != g.winner.name:
                st.session_state[g.slot] = override
                switch_winner(g)
            t.markdown(
                f"Outcome probability: {g.outcome_probability:.1%}"
                f"\n\nBrier: {_brier(g.outcome_probability, 1):.5f}"
            )
            t.divider()
            _set_next_games(game=g, games=tournament.games)

    current_r += 1
try:
    w_name = tournament.results["R6CH"].name
except KeyError as e:
    if e.args[0] == "R6CH":
        st.warning(
            "2025 tournament seeds are not yet available. "
            "Check back on March 17th when seeds have been assigned."
        )
        quit()
    else:
        raise e
ts.write(f"**{w_name} wins the tournament!**")

# summary
odds = np.array([g.outcome_probability for g in tournament.games if g.r != 0])
bracket_odds = int(1 / np.cumprod(odds)[-1])
avgbrier = _brier(odds, np.ones(len(odds))).mean()
success = (odds > 0.5).sum() / len(odds)

ts.markdown(
    """
        According to these probabilities, your odds of a perfect bracket
        based on these selections are 1 in **{a:,d}**... Yikes! Good luck!
        :) \n\n The brier score of this bracket outcome is {brier}
        with a model accuracy of {b:,d}%. These statistics do not include
        play-in games.
        """.format(
        a=bracket_odds, brier=round(avgbrier * 1e5) / 1e5, b=int(success * 100)
    )
)


def get_table_download_link():
    """
    Generates a link allowing the data in a given panda dataframe to be
    downloaded
    in:  dataframe
    out: href string
    """

    games = tournament.games
    slots = [g.slot for g in games]
    strong_name = [g.strong_team.name for g in games]
    weak_name = [g.weak_team.name for g in games]
    winning_name = [g.winner.name for g in games]
    df = pd.DataFrame.from_dict({"slot": slots})
    df.set_index("slot", inplace=True)
    df["strong_name"] = strong_name
    df["weak_name"] = weak_name
    df["winner"] = winning_name
    df["likelihood"] = np.array([g.outcome_probability for g in tournament.games])
    df["brier"] = _brier(df["likelihood"], np.ones(len(df["likelihood"])))

    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download results csv</a>'
    return href


ts.markdown(get_table_download_link(), unsafe_allow_html=True)


# Graph results
if ts.checkbox("Show graphical bracket - will slow app when checked"):
    ts.subheader("Sweet 16 and earlier")
    graph1 = graph_games(tournament, list(range(5)))
    ts.graphviz_chart(graph1)

    ts.subheader("Elite 8 and on")
    graph2 = graph_games(tournament, list(range(5, 7)))
    ts.graphviz_chart(graph2)

ts.divider()

if ts.checkbox("Show Game Result Dictionary with IDs (developer)"):
    ts.write({slot: t.id for slot, t in tournament.results.items()})

if ts.checkbox("Show Streamlit State (developer)"):
    ts.write(st.session_state)
