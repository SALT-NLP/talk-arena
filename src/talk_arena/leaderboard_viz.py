import json
import random
import textwrap
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from apscheduler.schedulers.background import BackgroundScheduler
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
WR_PLOT = None
BT_PLOT = None
UPDATE_TIME = None
NAME_MAPPING = {
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
    min_samp = melted_bootstrap["BT"].min()
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


def get_wr_plot():
    jrep = json.loads(pio.to_json(WR_PLOT))
    for step in jrep["layout"]["sliders"][0]["steps"]:
        step["args"][1]["frame"]["duration"] = 500
        step["args"][1]["transition"]["duration"] = 500
    jrep["layout"]["updatemenus"] = []
    jrep["layout"]["sliders"][0]["len"] = 0.8
    jrep["layout"]["sliders"][0]["pad"] = {}
    return json.dumps(jrep)


def get_bt_plot():
    jrep = json.loads(pio.to_json(BT_PLOT))
    for step in jrep["layout"]["sliders"][0]["steps"]:
        step["args"][1]["frame"]["duration"] = 500
        step["args"][1]["transition"]["duration"] = 500
    jrep["layout"]["updatemenus"] = []
    jrep["layout"]["sliders"][0]["len"] = 0.8
    jrep["layout"]["sliders"][0]["pad"] = {}
    return json.dumps(jrep)


def get_update_time():
    return UPDATE_TIME


def viz_factory(force=False):
    def process_and_visualize():
        """Main function to process JSON data and create visualizations."""
        global WR_PLOT, BT_PLOT, UPDATE_TIME
        if WR_PLOT is not None and BT_PLOT is not None and not force:
            return WR_PLOT, BT_PLOT, UPDATE_TIME
        try:
            # Read JSON data
            pub_json_data = open("/home/wheld3/talk-arena/live_votes.json", "r").read()
            prolific_json_data = open("/home/wheld3/talk-arena/prolific_votes.json", "r").read()
            merged_json_data = json.dumps(
                {"_default": {**json.loads(pub_json_data)["_default"], **json.loads(prolific_json_data)["_default"]}}
            )
            # Calculate win rates and create win rate plot
            pub_win_rates, pub_votes = calculate_win_rates(pub_json_data)
            pro_win_rates, pro_votes = calculate_win_rates(prolific_json_data)
            total_win_rates, total_votes = calculate_win_rates(merged_json_data)
            win_rates = (
                pd.concat([pub_win_rates, pro_win_rates, total_win_rates], keys=["Public", "Prolific", "Total"])
                .reset_index()
                .rename(columns={"level_0": "Source"})
            )
            WR_PLOT = create_win_rate_plot(win_rates)

            # Calculate Bradley-Terry ratings and create BT plot
            pub_bootstrap_ratings = compute_bootstrap_bt(pub_json_data, num_round=10000)
            pro_bootstrap_ratings = compute_bootstrap_bt(prolific_json_data, num_round=10000)
            total_bootstrap_ratings = compute_bootstrap_bt(merged_json_data, num_round=10000)
            bootstrap_ratings = (
                pd.concat(
                    [pub_bootstrap_ratings, pro_bootstrap_ratings, total_bootstrap_ratings],
                    keys=["Public", "Prolific", "Total"],
                )
                .reset_index()
                .rename(columns={"level_0": "Source"})
            )
            BT_PLOT = create_bt_plot(bootstrap_ratings)
            UPDATE_TIME = gr.Markdown(
                value=textwrap.dedent(
                    f"""
                    <h4 class="nx-font-semibold nx-tracking-tight nx-text-slate-900 dark:nx-text-slate-100 nx-text-xl">Last Refresh: {get_aesthetic_timestamp()} PST</h4>
                    <h6 class="nx-font-semibold nx-tracking-tight nx-text-slate-900 dark:nx-text-slate-100nx-text-base">Total Votes: {total_votes}, Public Votes: {pub_votes}, Prolific Votes: {pro_votes}</h6>
                    """
                )
            )
            return WR_PLOT, BT_PLOT, UPDATE_TIME

        except Exception as e:
            raise gr.Error(f"Error processing file: {str(e)}")

    return process_and_visualize


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)

# Create Gradio interface
with gr.Blocks(title="Talk Arena Leaderboard Analysis", theme=theme) as demo:
    viz_factory(force=True)()
    last_updated = UPDATE_TIME
    with gr.Row():
        bt_plot = gr.Plot(label="Bradley-Terry Ratings", value=BT_PLOT)
    with gr.Row():
        win_rate_plot = gr.Plot(label="Win Rates", value=WR_PLOT)

    d1 = gr.Textbox(visible=False)
    demo.load(
        fn=viz_factory(force=False), inputs=[], outputs=[win_rate_plot, bt_plot, last_updated], show_progress="minimal"
    )
    demo.load(fn=get_wr_plot, inputs=[], outputs=[d1])
    demo.load(fn=get_bt_plot, inputs=[], outputs=[d1])
    demo.load(fn=get_update_time, inputs=[], outputs=[d1])

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=viz_factory(force=True), trigger="interval", seconds=300)
    scheduler.start()
    demo.queue(default_concurrency_limit=10, api_open=True).launch(share=True, server_port=8004, node_port=8005)
