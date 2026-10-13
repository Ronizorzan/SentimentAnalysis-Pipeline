import json
import math
import yaml
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Categorical, Real
import mlflow
from mlflow.sklearn import log_model

from core.utils import elapsed_time, logger


with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RANDOM_SEED = params["seed"]["random_state"]

@elapsed_time
def load_and_split_data(file_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(file_path)
    X = data["text"]
    y = data["sentiments"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["search"]["test_size"],
        random_state=RANDOM_SEED,
        stratify=y,
    )
    train_data = pd.DataFrame({"text": X_train, "sentiments": y_train})
    test_data = pd.DataFrame({"text": X_test, "sentiments": y_test})

    train_path = Path(params["artifacts"]["train_split"])
    test_path = Path(params["artifacts"]["test_split"])
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_data.to_parquet(train_path, engine="pyarrow", index=False)
    test_data.to_parquet(test_path, engine="pyarrow", index=False)

    return train_data, test_data


def build_bayesian_search(
    search_config: dict[str, Any], random_seed: int
) -> BayesSearchCV:
    pipeline = Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
        ),
        ("classifier", LogisticRegression(max_iter=1000, random_state=random_seed)),
    ])
    search_space = {
        "vectorizer__sublinear_tf": Categorical(search_config["sublinear_tf_values"]),
        "classifier__C": Real(
            search_config["c_value_min"],
            search_config["c_value_max"],
            prior="log-uniform",
        ),
        "classifier__class_weight": Categorical(search_config["class_weights"]),
    }
    cv = StratifiedKFold(
        n_splits=search_config["cv_folds"],
        shuffle=True,
        random_state=random_seed,
    )
    return BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        n_iter=search_config["n_iter"],
        scoring=search_config["scoring"],
        cv=cv,
        n_jobs=search_config["n_jobs"],
        random_state=random_seed,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )


def evaluate_model(estimator: Any, test_data: pd.DataFrame) -> dict[str, Any]:
    predictions = estimator.predict(test_data["text"])
    return {
        "accuracy": accuracy_score(test_data["sentiments"], predictions),
        "classification_report": classification_report(
            test_data["sentiments"], predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            test_data["sentiments"], predictions
        ).tolist(),
    }


def to_json_safe(value: Any) -> Any:
    if np.ma.is_masked(value):
        return None
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(to_json_safe(payload), output_file, ensure_ascii=False, indent=2, allow_nan=False)


def log_mlflow_run(
    search: BayesSearchCV,
    search_summary: dict[str, Any],
    holdout_metrics: dict[str, Any],
) -> None:
    tracking = params["tracking"]
    if not tracking["enabled"]:
        return

    
    mlflow.set_tracking_uri(tracking["tracking_uri"])
    mlflow.set_experiment(tracking["experiment_name"])
    with mlflow.start_run(run_name="bayesian-logistic-regression-search"):
        mlflow.log_params({
            "random_seed": RANDOM_SEED,
            "n_iter": params["search"]["n_iter"],
            "cv_folds": params["search"]["cv_folds"],
            "scoring": params["search"]["scoring"],
            "search_config": json.dumps(to_json_safe(params["search"])),
            **{
                f"best.{key}": json.dumps(to_json_safe(value))
                for key, value in search.best_params_.items()
            },
        })
        mlflow.log_metrics({
            "cv_best_score": float(search.best_score_),
            "holdout_accuracy": float(holdout_metrics["accuracy"]),
        })
        mlflow.log_dict(search_summary, "bayes_search_summary.json")
        mlflow.log_dict(holdout_metrics, "holdout_metrics.json")
        log_model(search.best_estimator_, name="best-model")


@elapsed_time
def train_model_with_cross_validation(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    search_config: dict[str, Any] | None = None,
) -> BayesSearchCV:
    if search_config is None:
        resolved_search_config = params["search"]
    else:
        resolved_search_config = search_config
    search = build_bayesian_search(resolved_search_config, RANDOM_SEED)
    search.fit(train_data["text"], train_data["sentiments"])

    holdout_metrics = evaluate_model(search.best_estimator_, test_data)
    search_summary = {
        "random_seed": RANDOM_SEED,
        "search_config": to_json_safe(resolved_search_config),
        "scoring": resolved_search_config["scoring"],
        "cv_folds": resolved_search_config["cv_folds"],
        "n_iter": resolved_search_config["n_iter"],
        "best_params": to_json_safe(search.best_params_),
        "best_cv_score": float(search.best_score_),
        "holdout_accuracy": float(holdout_metrics["accuracy"]),
    }
    cv_results = pd.DataFrame(search.cv_results_).to_dict(orient="records")
    outputs = params["artifacts"]
    write_json(outputs["search_summary"], search_summary)
    write_json(outputs["cv_results"], {"results": cv_results})
    write_json(outputs["holdout_metrics"], holdout_metrics)

    model_path = Path(outputs["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, model_path)
    log_mlflow_run(search, search_summary, holdout_metrics)

    logger.info(f"Melhor score de CV ({resolved_search_config['scoring']}): {search.best_score_:.4f}")
    logger.info(f"Melhores hiperparâmetros: {search.best_params_}")
    logger.info(f"Acurácia no holdout: {holdout_metrics['accuracy']:.4f}")
    logger.info(f"Matriz de confusão do holdout: {holdout_metrics['confusion_matrix']}")
    return search




def main():
    file_path = "processed_data/clean_tweets_full.parquet"
    train_data, test_data = load_and_split_data(file_path)
    train_model_with_cross_validation(train_data, test_data)
    logger.info("Treinamento Concluído com sucesso")


if __name__ == "__main__":
    main()



