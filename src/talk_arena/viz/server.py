import hashlib
import json
import textwrap
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import plotly.io as pio
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from talk_arena.viz.core import *

app = FastAPI(title="Talk Arena API", description="API for Talk Arena leaderboard and statistics", version="0.0.1")

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
    LAST_PROCESSED = None
    MIN_UPDATE_INTERVAL = 60  # Minimum seconds between updates


state = GlobalState()


def process_and_visualize(force: bool = False):
    """Process data and create visualizations"""
    global state
    current_time = datetime.now(ZoneInfo("America/Los_Angeles"))

    # Check if enough time has passed since last update
    if not force and state.LAST_PROCESSED:
        time_diff = (current_time - state.LAST_PROCESSED).total_seconds()
        if time_diff < state.MIN_UPDATE_INTERVAL:
            logger.info(f"Skipping update - only {time_diff:.1f} seconds since last update")
            return

    state.LAST_PROCESSED = current_time
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

        win_rates = (
            pd.concat([pub_win_rates, pro_win_rates, total_win_rates], keys=["Public", "Prolific", "Total"])
            .reset_index()
            .rename(columns={"level_0": "Source"})
        )

        state.WR_PLOT = create_win_rate_plot(win_rates)

        # Calculate Bradley-Terry ratings
        pub_bootstrap_ratings = compute_bootstrap_bt(pub_json_data, num_round=10000)
        pro_bootstrap_ratings = compute_bootstrap_bt(prolific_json_data, num_round=10000)
        total_bootstrap_ratings = compute_bootstrap_bt(merged_json_data, num_round=10000)

        for model in all_models:
            if model not in pro_models:
                pro_bootstrap_ratings[model] = pro_bootstrap_ratings["diva_3_8b"] * -1

        bootstrap_ratings = (
            pd.concat(
                [pub_bootstrap_ratings, pro_bootstrap_ratings, total_bootstrap_ratings],
                keys=["Public", "Prolific", "Total"],
            )
            .reset_index()
            .rename(columns={"level_0": "Source"})
        )

        state.BT_PLOT = create_bt_plot(bootstrap_ratings)

        # Update timestamp and vote counts
        state.UPDATE_TIME = {
            "timestamp": get_aesthetic_timestamp(),
            "total_votes": total_votes,
            "public_votes": pub_votes,
            "prolific_votes": pro_votes,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


# Set up logging
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = None


def update_job():
    """Wrapper for the update job with error handling and logging"""
    try:
        logger.info("Starting scheduled update...")
        process_and_visualize(force=True)
        logger.info("Scheduled update completed successfully")
    except Exception as e:
        logger.error(f"Error in scheduled update: {str(e)}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    """Initialize data and start scheduler"""
    global scheduler

    try:
        logger.info("Starting initial data processing...")
        process_and_visualize(force=True)
        logger.info("Initial data processing completed")

        # Clear any existing schedulers
        if scheduler:
            scheduler.shutdown(wait=False)

        # Initialize and start the scheduler
        scheduler = BackgroundScheduler(
            timezone=ZoneInfo("America/Los_Angeles"), job_defaults={"coalesce": True, "max_instances": 1}
        )

        # Add the job with proper error handling
        scheduler.add_job(
            func=update_job,  # Use the wrapper function
            trigger="interval",
            seconds=300,
            id="update_visualizations",
            name="Update Visualizations",
            misfire_grace_time=60,
        )

        scheduler.start()
        logger.info("Scheduler started successfully")

        # Verify the job was added
        jobs = scheduler.get_jobs()
        logger.info(f"Current scheduled jobs: {[job.name for job in jobs]}")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Properly shutdown the scheduler when the app stops"""
    global scheduler
    try:
        if scheduler:
            logger.info("Shutting down scheduler...")
            scheduler.shutdown(wait=False)
            logger.info("Scheduler shutdown complete")
    except Exception as e:
        logger.error(f"Error during scheduler shutdown: {str(e)}", exc_info=True)


# Add an endpoint to manually trigger an update
@app.post("/api/trigger-update")
async def trigger_update():
    """Manually trigger a data update"""
    try:
        logger.info("Manual update triggered")
        process_and_visualize(force=True)
        logger.info("Manual update completed")
        return {"status": "success", "message": "Update completed"}
    except Exception as e:
        logger.error(f"Error in manual update: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def generate_etag(data: dict) -> str:
    """Generate an ETag for the given data"""
    # Convert data to a consistent string representation and hash it
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


@app.get("/api/win-rate-plot")
async def get_wr_plot(response: Response):
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

    # Generate ETag
    etag = generate_etag(plot_json)
    response.headers["ETag"] = etag

    # Set cache control headers - cache for 4 minutes since we update every 5
    response.headers["Cache-Control"] = "public, max-age=240"

    return plot_json


@app.get("/api/bt-plot")
async def get_bt_plot(response: Response):
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

    # Generate ETag
    etag = generate_etag(plot_json)
    response.headers["ETag"] = etag

    # Set cache control headers - cache for 4 minutes since we update every 5
    response.headers["Cache-Control"] = "public, max-age=240"

    return plot_json


@app.get("/api/update-time")
async def get_update_time(response: Response):
    """Get the last update time and vote counts"""
    if state.UPDATE_TIME is None:
        raise HTTPException(status_code=503, detail="Update time not yet available")

    # Generate ETag
    etag = generate_etag(state.UPDATE_TIME)
    response.headers["ETag"] = etag

    # Set cache control headers - cache for 4 minutes
    response.headers["Cache-Control"] = "public, max-age=240"

    return state.UPDATE_TIME


@app.get("/api/health")
async def health_check(response: Response):
    """Enhanced health check endpoint with scheduler status"""
    global scheduler

    scheduler_status = "not_running"
    next_run = None
    last_run = state.UPDATE_TIME["timestamp"] if state.UPDATE_TIME else None

    if scheduler:
        try:
            jobs = scheduler.get_jobs()
            if jobs:
                scheduler_status = "running"
                next_run = jobs[0].next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception as e:
            logger.error(f"Error checking scheduler status: {str(e)}")
            scheduler_status = f"error: {str(e)}"

    health_data = {
        "status": "healthy",
        "scheduler_status": scheduler_status,
        "last_update": last_run,
        "next_scheduled_update": next_run,
        "current_time": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

    # Generate ETag
    etag = generate_etag(health_data)
    response.headers["ETag"] = etag

    # Set cache control headers - short cache time for health check
    response.headers["Cache-Control"] = "public, max-age=30"

    return health_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
