import mlflow
import random

def run_analysis():
    mean = random.uniform(3.5, 4.5)
    std = random.uniform(0.8, 1.2)

    with mlflow.start_run():
        mlflow.log_metric("mean", mean)
        mlflow.log_metric("std", std)
        mlflow.log_dict(
            {
                "mean": mean,
                "std": std
            },
            "result.json"
        )

    return {
        "mean": mean,
        "std": std
    }