from pathlib import Path

import pandas as pd
from models.RandomForestClassifier import train_RandomForestClassifier
from models.svm import train_svm
from models.xgboost_regressor import train_XGBRegressor
from utils.artifacts import (
    build_run_info,
    create_run_dir,
    save_json,
    save_model,
)
from utils.evaluation import evaluate_model
from utils.preprocessing import split_data

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "stroke_classification.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODELS = {
    "svm": train_svm,
    "RandomForest": train_RandomForestClassifier,
    "xgboost": train_XGBRegressor,
}




def main():
    data = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = split_data(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    run_dir = create_run_dir(BASE_DIR)
    save_json(
        build_run_info(
            DATA_PATH,
            data,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            models=list(MODELS),
        ),
        run_dir / "run_info.json",
    )

    summary = []
    for name, train_fn in MODELS.items():
        print(f"Training {name}...")
        model = train_fn(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)

        save_model(model, run_dir / "model" / f"{name}.joblib")
        save_json(metrics, run_dir / f"{name}_metrics.json")

        summary.append({"model": name, **metrics})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(run_dir/"summary.csv", index=False)
    shown = summary_df[
        ["model", "accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"]
    ]
    print(shown.round(3).to_string(index=False))
    print(f"Save to: {run_dir}")

if __name__ == "__main__":
    main()