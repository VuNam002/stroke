from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from utils.preprocessing import build_preprocessing_pipeline
from utils.tuning import grid_search


def train_svm(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", SVC(class_weight="balanced", cache_size=500, random_state=42)),
    ])
    param_grid = [
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.01, 0.1],
        },
        {
            "model__kernel": ["linear"],
            "model__C": [0.1, 1, 10],
        },
    ]
    return grid_search(reg, param_grid, x_train, y_train, scoring="average_precision", n_jobs=8)
