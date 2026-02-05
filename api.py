from fastapi import FastAPI, Header, HTTPException, Depends
import mlflow
import json
import os

from analysis_logic import run_analysis

app = FastAPI()


# -------------------------
# API key protection
# -------------------------
def require_api_key(x_api_key: str = Header(...)):
    expected_key = os.environ.get("API_KEY")

    if expected_key is None:
        raise HTTPException(status_code=500, detail="API key not configured")

    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# -------------------------
# MLflow helpers
# -------------------------
def get_latest_run_id():
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("Default")
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        return None

    return runs[0].info.run_id


# -------------------------
# API endpoints
# -------------------------
@app.post("/seed", dependencies=[Depends(require_api_key)])
def seed():
    result = run_analysis()
    return {
        "status": "seeded",
        "result": result
    }


@app.get("/summary", dependencies=[Depends(require_api_key)])
def summary():
    run_id = get_latest_run_id()

    if run_id is None:
        return {
            "status": "no_data",
            "message": "No MLflow runs found yet"
        }

    client = mlflow.tracking.MlflowClient()
    path = client.download_artifacts(run_id, "result.json")

    with open(path) as f:
        return json.load(f)