import json
import random
import textwrap
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from scipy.optimize import minimize
from scipy.special import expit

# Constants
COLORS = [
    "#1B7FFF",
    "#F07D1A",
    "#BA24C7",
    "#FE42C7",
    "#0D4B7C",
    "#0EAC96",
    "#AA7CFF",
    "#B50550",
    "#009EEB",
    "#220B55",
    "#7B3301",
]
NAME_MAPPING = {
    "gemini_2f": "Gemini 2.0 (Exp)",
    "diva_3_8b": "DiVA Llama 3 8B",
    "qwen2": "Qwen 2 Audio",
    "pipe_l3.0": "Pipelined Llama 3 8B",
    "gemini_1.5f": "Gemini 1.5 Flash",
    "gpt4o": "GPT-4o",
    "gemini_1.5p": "Gemini 1.5 Pro",
    "typhoon_audio": "Typhoon Audio",
}

def get_aesthetic_timestamp():
    """
    Returns a beautifully formatted timestamp in the format:
    'Tuesday, December 10th, 2024 at 3:45 PM'
    """
    # Get timezone object for PST
    pst = ZoneInfo("America/Los_Angeles")

    # Get current time in PST
    now = datetime.now(pst)

    # Add suffix to day number (1st, 2nd, 3rd, etc.)
    day = now.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    return now.strftime(f"%A, %B {day}{suffix}, %Y at %-I:%M %p")


def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """Calculate bootstrap confidence intervals."""
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        bootstrap_samples.append(np.mean(random.choices(data, k=len(data))))
    lower_bound = np.percentile(bootstrap_samples, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_samples, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound


def calculate_win_rates(json_data):
    """Calculate win rates from JSON data."""
    data = json.loads(json_data)

    model_wins = defaultdict(int)
    total_matches = defaultdict(int)
    total_votes = 0

    for value in data["_default"].values():
        total_votes += 1
        if value["outcome"] == 0:
            model_wins[value["model_a"]] += 1
        elif value["outcome"] == 1:
            model_wins[value["model_b"]] += 1
        elif value["outcome"] == 0.5:
            model_wins[value["model_a"]] += 0.5
            model_wins[value["model_b"]] += 0.5
        total_matches[value["model_a"]] += 1
        total_matches[value["model_b"]] += 1

    per_model_wins = {}
    for model, wins in model_wins.items():
        win_rate = wins / total_matches[model]
        wins_data = [1] * int(wins) + [0] * int(total_matches[model] - wins)
        if int(wins) != wins:
            wins_data += [0.5]
        lower, upper = bootstrap_ci(wins_data)
        per_model_wins[model] = {
            "model": model,
            "win_rate": win_rate,
            "95_lower": (win_rate - lower),
            "95_upper": (upper - win_rate),
        }
    df = pd.DataFrame.from_dict(per_model_wins).T

    return df, total_votes


def create_win_rate_plot(wins_df):
    """Create win rate plot using Plotly."""
    wins_df["Source"] = wins_df["Source"].astype(str)
    wins_df = wins_df.sort_values(by=["Source", "win_rate"], ascending=False)
    wins_df["model"] = wins_df["model"].apply(lambda x: NAME_MAPPING.get(x, x))

    fig = px.bar(
        wins_df,
        x="model",
        y="win_rate",
        error_y="95_upper",
        error_y_minus="95_lower",
        color="model",
        color_discrete_sequence=COLORS,
        animation_group="model",
        animation_frame="Source",
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" + "Win Rate: %{y}" + "<extra></extra>",
    )

    fig.update_layout(
        autosize=True,
        showlegend=False,
        plot_bgcolor="white",
        title={
            "text": "Talk Arena Live Win Rates<br>with 95% Confidence Intervals",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Model",
        yaxis_title="Win Rate (%)",
        bargap=0.2,
        yaxis=dict(
            tickformat=".0%", tickmode="auto", range=[0, 1.01], gridcolor="#C9CCD1", griddash="dash", gridwidth=2
        ),
        legend=dict(
            orientation="h",  # Make legend horizontal
            yanchor="bottom",
            y=-0.5,  # Position below plot
            xanchor="center",
            x=0.5,  # Center horizontally
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#C9CCD1",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=0, b=10),  # Balanced margins
        hoverlabel=dict(bgcolor="white", font_size=14, bordercolor="gray"),
    )

    fig.update_xaxes(showgrid=False)

    return fig


# Bradley-Terry Model Functions
def load_live_votes(json_str: str) -> pd.DataFrame:
    """Load and preprocess live votes data from JSON string."""
    data = json.loads(json_str)
    df = pd.DataFrame.from_dict(data["_default"], orient="index")
    df["winner"] = df["outcome"].map({1: "model_b", 0: "model_a", 0.5: "tie"})
    return df


def preprocess_for_bt(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Preprocess data for Bradley-Terry model fitting."""
    all_models = pd.concat([df["model_a"], df["model_b"]]).unique()
    model_to_idx = {model: idx for idx, model in enumerate(all_models)}

    matchups = np.array([[model_to_idx[row.model_a], model_to_idx[row.model_b]] for _, row in df.iterrows()])

    outcomes = np.array(
        [1.0 if row.winner == "model_a" else (0.5 if row.winner == "tie" else 0.0) for _, row in df.iterrows()]
    )

    unique_matches = np.column_stack([matchups, outcomes])
    unique_matches, weights = np.unique(unique_matches, return_counts=True, axis=0)

    return (unique_matches[:, :2].astype(np.int32), unique_matches[:, 2], list(all_models), weights.astype(np.float64))


def bt_loss_and_grad(
    ratings: np.ndarray, matchups: np.ndarray, outcomes: np.ndarray, weights: np.ndarray, alpha: float = 1.0
) -> Tuple[float, np.ndarray]:
    """Compute Bradley-Terry loss and gradient."""
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    probs = expit(logits)

    loss = -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights).sum()

    matchups_grads = -alpha * (outcomes - probs) * weights
    model_grad = np.zeros_like(ratings)
    np.add.at(model_grad, matchups[:, [0, 1]], matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64))

    return loss, model_grad


def fit_bt(
    matchups: np.ndarray, outcomes: np.ndarray, weights: np.ndarray, n_models: int, alpha: float, tol: float = 1e-6
) -> np.ndarray:
    """Fit Bradley-Terry model using L-BFGS-B optimization."""
    initial_ratings = np.zeros(n_models, dtype=np.float64)

    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )

    return result["x"]


def scale_and_offset(
    ratings: np.ndarray, models: List[str], scale: float = 400, init_rating: float = 1000
) -> np.ndarray:
    """Scale ratings to familiar Elo-like scale."""
    scaled_ratings = (ratings * scale) + init_rating
    return scaled_ratings


def compute_bootstrap_bt(
    data: str,
    num_round: int = 100,
    base: float = 10.0,
    scale: float = 400.0,
    init_rating: float = 1000.0,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Compute bootstrap Bradley-Terry ratings from live votes data."""
    df = load_live_votes(data)
    matchups, outcomes, models, weights = preprocess_for_bt(df)

    rng = np.random.default_rng(seed=0)
    total_matches = len(df)
    idxs = rng.multinomial(n=total_matches, pvals=weights / weights.sum(), size=num_round)
    boot_weights = idxs.astype(np.float64) / total_matches

    ratings_list = []
    for sample_weights in boot_weights:
        ratings = fit_bt(
            matchups=matchups,
            outcomes=outcomes,
            weights=sample_weights,
            n_models=len(models),
            alpha=np.log(base),
            tol=tol,
        )
        scaled_ratings = scale_and_offset(ratings=ratings, models=models, scale=scale, init_rating=init_rating)
        ratings_list.append(scaled_ratings)

    df_ratings = pd.DataFrame(ratings_list, columns=models)
    return df_ratings[df_ratings.median().sort_values(ascending=False).index]


def create_bt_plot(bootstrap_ratings):
    """Create Bradley-Terry ratings plot using Plotly."""
    melted_bootstrap = bootstrap_ratings.melt(id_vars=["Source", "level_1"], var_name="Model", value_name="BT")
    melted_bootstrap = melted_bootstrap.dropna()
    melted_bootstrap = melted_bootstrap.sort_values(by=["Source", "Model", "BT"], ascending=False)
    # Pretty Names
    melted_bootstrap["Model"] = melted_bootstrap["Model"].apply(lambda x: NAME_MAPPING.get(x, x))
    # Compression for Client Side
    melted_bootstrap["BT"] = melted_bootstrap["BT"].apply(lambda x: int(x))
    min_samp = melted_bootstrap[melted_bootstrap["BT"] > 0]["BT"].min()
    max_samp = melted_bootstrap["BT"].max()
    idx_keep = list(range(0, len(melted_bootstrap), 10))
    melted_bootstrap = melted_bootstrap.iloc[idx_keep]
    melted_bootstrap = melted_bootstrap.sort_values(by=["Source", "BT"], ascending=False)
    fig = px.violin(
        melted_bootstrap,
        x="Model",
        y="BT",
        color="Model",
        animation_group="Model",
        animation_frame="Source",
        color_discrete_sequence=COLORS,
    )

    fig.update_layout(
        autosize=True,
        showlegend=False,
        plot_bgcolor="white",
        title={
            "text": "Talk Arena Live Bradley-Terry Ratings<br>with Bootstrapped Variance",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Model",
        yaxis_title="Rating",
        yaxis=dict(gridcolor="#C9CCD1", range=[min_samp - 10, max_samp + 10], griddash="dash"),
        legend=dict(
            orientation="h",  # Make legend horizontal
            yanchor="bottom",
            y=-0.5,  # Position below plot
            xanchor="center",
            x=0.5,  # Center horizontally
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#C9CCD1",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=0, b=10),  # Balanced margins
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=2)

    return fig
