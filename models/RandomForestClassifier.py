from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from utils.preprocessing import build_preprocessing_pipeline
from utils.tuning import grid_search


def train_RandomForestClassifier(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)),
    ])
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 5, 10],
        "model__max_features": ["sqrt", 0.5],
    }
    return grid_search(reg, param_grid, x_train, y_train, scoring="average_precision", n_jobs=8)
