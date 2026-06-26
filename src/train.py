import pickle
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from preprocess import run_preprocessing


# 1. MLFLOW SETUP

# An "experiment" in MLflow is a named container for related runs
# All our churn model runs will live under this experiment name
EXPERIMENT_NAME = "churn-prediction"

mlflow.set_experiment(EXPERIMENT_NAME)


# 2. EVALUATION FUNCTION

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of churn (class=1)

    metrics = {
        "auc_roc":   roc_auc_score(y_test, y_proba),
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
    }

    return metrics


# 3. SINGLE TRAINING RUN

def train_and_log(
    X_train, X_test, y_train, y_test,
    n_estimators: int = 100,
    max_depth: int = None,
    class_weight: str = "balanced",
    run_name: str = "default-run"
):

    # Every block inside "with mlflow.start_run()" is automatically recorded
    with mlflow.start_run(run_name=run_name):

        params = {
            "n_estimators":  n_estimators,
            "max_depth":     max_depth,
            "class_weight":  class_weight,
            "random_state":  67
        }
        mlflow.log_params(params)

        print(f"\nTraining run: {run_name}")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=67,
            n_jobs=-1          # use all CPU cores
        )
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

        # This saves the model inside the MLflow run so you can load it later
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random-forest-model",
            registered_model_name="churn-rf-model"
        )

        run_id = mlflow.active_run().info.run_id

    return model, metrics, run_id


# 4. EXPERIMENT: COMPARE MULTIPLE CONFIGURATIONS

def run_experiments(data_path: str):

    print("Running preprocessing...")
    _, X_train, X_test, y_train, y_test = run_preprocessing(data_path)

    # Definition of the configurations we want to compare
    configs = [
        {
            "run_name":     "rf-100trees-no-depth-limit",
            "n_estimators": 100,
            "max_depth":    None,   # None = trees grow until pure leaves
        },
        {
            "run_name":     "rf-200trees-no-depth-limit",
            "n_estimators": 200,
            "max_depth":    None,
        },
        {
            "run_name":     "rf-100trees-depth10",
            "n_estimators": 100,
            "max_depth":    10,     # shallower trees = less overfitting
        },
        {
            "run_name":     "rf-200trees-depth10",
            "n_estimators": 200,
            "max_depth":    10,
        },
    ]

    results = []

    for config in configs:
        model, metrics, run_id = train_and_log(
            X_train, X_test, y_train, y_test,
            **config
        )
        results.append({
            "run_name": config["run_name"],
            "run_id":   run_id,
            "auc_roc":  metrics["auc_roc"],
            "f1":       metrics["f1"],
            "model":    model
        })

    # Pick the best model by AUC-ROC
    best = max(results, key=lambda x: x["auc_roc"])

    print(f"\n{'='*50}")
    print(f"Best run: {best['run_name']}")
    print(f"Best AUC-ROC: {best['auc_roc']:.4f}")
    print(f"Run ID: {best['run_id']}")

    # Save the best model locally for FastAPI to load
    # (in addition to the MLflow registry)
    with open("artifacts/best_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)

    print("\nBest model saved to artifacts/best_model.pkl")
    return best


if __name__ == "__main__":
    run_experiments("data/Telco-Customer-Churn.csv")
