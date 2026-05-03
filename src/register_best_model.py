import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "quickfoods-delivery-predictor"

def main():
    client = MlflowClient()

    # Get all runs from experiment
    experiment = mlflow.get_experiment_by_name("quickfoods-delivery-time")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Find best run (lowest MAE)
    best_run = runs.sort_values("metrics.mae").iloc[0]

    best_run_id = best_run.run_id
    best_mae = best_run["metrics.mae"]

    print("\n=== QuickFoods: Promote Best Tuned Model ===")
    print(f"Best run ID : {best_run_id}")
    print(f"Best MAE    : {best_mae}")

    model_uri = f"runs:/{best_run_id}/model"

    # Register model
    result = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"\nRegistered '{MODEL_NAME}' version {result.version}")
    print("Status: READY")

if __name__ == "__main__":
    main()