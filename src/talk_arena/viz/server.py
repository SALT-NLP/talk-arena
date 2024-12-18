import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import json
import plotly.io as pio
from typing import Optional
from datetime import datetime
import textwrap
from zoneinfo import ZoneInfo
from talk_arena.viz.core import *

app = FastAPI(
    title="Talk Arena API",
    description="API for Talk Arena leaderboard and statistics",
    version="0.0.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the plots and update time
class GlobalState:
    WR_PLOT = None
    BT_PLOT = None
    UPDATE_TIME = None

state = GlobalState()

def process_and_visualize(force: bool = False):
    """Process data and create visualizations"""
    if state.WR_PLOT is not None and state.BT_PLOT is not None and not force:
        return
    
    try:
        # Read JSON data
        pub_json_data = open("/home/wheld3/talk-arena/live_votes.json", "r").read()
        prolific_json_data = open("/home/wheld3/talk-arena/prolific_votes.json", "r").read()
        merged_json_data = json.dumps(
            {"_default": {**json.loads(pub_json_data)["_default"], **json.loads(prolific_json_data)["_default"]}}
        )

        # Calculate win rates and create plots
        pub_win_rates, pub_votes = calculate_win_rates(pub_json_data)
        pro_win_rates, pro_votes = calculate_win_rates(prolific_json_data)
        total_win_rates, total_votes = calculate_win_rates(merged_json_data)

        # Process win rates
        all_models = total_win_rates["model"].unique()
        pro_models = pro_win_rates["model"].unique()
        for model in all_models:
            if model not in pro_models:
                new_index = len(pro_win_rates)
                pro_win_rates.loc[new_index] = [model, -0.1, -0.1, -0.2]

        win_rates = pd.concat(
            [pub_win_rates, pro_win_rates, total_win_rates],
            keys=["Public", "Prolific", "Total"]
        ).reset_index().rename(columns={"level_0": "Source"})

        state.WR_PLOT = create_win_rate_plot(win_rates)

        # Calculate Bradley-Terry ratings
        pub_bootstrap_ratings = compute_bootstrap_bt(pub_json_data, num_round=10000)
        pro_bootstrap_ratings = compute_bootstrap_bt(prolific_json_data, num_round=10000)
        total_bootstrap_ratings = compute_bootstrap_bt(merged_json_data, num_round=10000)

        for model in all_models:
            if model not in pro_models:
                pro_bootstrap_ratings[model] = pro_bootstrap_ratings["diva_3_8b"] * -1

        bootstrap_ratings = pd.concat(
            [pub_bootstrap_ratings, pro_bootstrap_ratings, total_bootstrap_ratings],
            keys=["Public", "Prolific", "Total"]
        ).reset_index().rename(columns={"level_0": "Source"})

        state.BT_PLOT = create_bt_plot(bootstrap_ratings)
        
        # Update timestamp and vote counts
        state.UPDATE_TIME = {
            "timestamp": get_aesthetic_timestamp(),
            "total_votes": total_votes,
            "public_votes": pub_votes,
            "prolific_votes": pro_votes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize data and start scheduler"""
    global scheduler
    
    # Initial data processing
    process_and_visualize(force=True)
    
    # Initialize and start the scheduler
    scheduler = BackgroundScheduler()
    
    scheduler.add_job(
        func=process_and_visualize,
        trigger="interval",
        seconds=300,  # 5 minutes
        kwargs={"force": True},
        id="update_visualizations",
        name="Update Visualizations",
    )
    
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Properly shutdown the scheduler when the app stops"""
    global scheduler
    if scheduler:
        scheduler.shutdown(wait=False)

@app.get("/api/win-rate-plot")
async def get_wr_plot():
    """Get the win rate plot data"""
    if state.WR_PLOT is None:
        raise HTTPException(status_code=503, detail="Plot data not yet available")
    
    plot_json = json.loads(pio.to_json(state.WR_PLOT))
    
    # Customize animation settings
    for step in plot_json["layout"]["sliders"][0]["steps"]:
        step["args"][1]["frame"]["duration"] = 500
        step["args"][1]["transition"]["duration"] = 500
    
    plot_json["layout"]["updatemenus"] = []
    plot_json["layout"]["sliders"][0]["len"] = 0.8
    plot_json["layout"]["sliders"][0]["pad"] = {}
    
    return plot_json

@app.get("/api/bt-plot")
async def get_bt_plot():
    """Get the Bradley-Terry plot data"""
    if state.BT_PLOT is None:
        raise HTTPException(status_code=503, detail="Plot data not yet available")
    
    plot_json = json.loads(pio.to_json(state.BT_PLOT))
    
    # Customize animation settings
    for step in plot_json["layout"]["sliders"][0]["steps"]:
        step["args"][1]["frame"]["duration"] = 500
        step["args"][1]["transition"]["duration"] = 500
    
    plot_json["layout"]["updatemenus"] = []
    plot_json["layout"]["sliders"][0]["len"] = 0.8
    plot_json["layout"]["sliders"][0]["pad"] = {}
    
    return plot_json

@app.get("/api/update-time")
async def get_update_time():
    """Get the last update time and vote counts"""
    if state.UPDATE_TIME is None:
        raise HTTPException(status_code=503, detail="Update time not yet available")
    return state.UPDATE_TIME

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Global scheduler instance
    scheduler = None
    uvicorn.run(app, host="0.0.0.0", port=8000)
