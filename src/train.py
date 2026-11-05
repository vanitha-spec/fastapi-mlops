import json
import os
import tempfile

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.feature_pipeline import build_preprocessor

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("churn-prediction")

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=6),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}


def train():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train, y_train = train_df.drop(columns=["Churn"]), train_df["Churn"]
    X_test, y_test = test_df.drop(columns=["Churn"]), test_df["Churn"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model, best_score, best_name = None, 0.0, None
    all_metrics = {}

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", model)])

            # the best model is chosen with cross-validation on TRAIN data only
            cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "precision": round(precision_score(y_test, preds), 4),
                "recall": round(recall_score(y_test, preds), 4),
                "f1": round(f1_score(y_test, preds), 4),
                "roc_auc": round(roc_auc_score(y_test, proba), 4),
                "cv_roc_auc": round(cv_auc, 4),
            }
            all_metrics[name] = metrics

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            # save this model as a file inside the MLflow run
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, f"{name}.pkl")
                joblib.dump(pipe, path)
                mlflow.log_artifact(path, artifact_path="model")

            print(f"{name}: {metrics}")

            if cv_auc > best_score:
                best_score, best_model, best_name = cv_auc, pipe, name

    # save the best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    with open("models/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": all_metrics}, f, indent=2)

    print(f"Best model: {best_name} (cv_auc={best_score:.4f})")
    print("Model saved to models/best_model.pkl")


if __name__ == "__main__":
    train()