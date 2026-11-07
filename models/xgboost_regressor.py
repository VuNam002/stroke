from sklearn.pipeline import Pipeline
from utils.preprocessing import build_preprocessing_pipeline
from utils.tuning import grid_search
from xgboost import XGBClassifier


def train_XGBRegressor(x_train, y_train):
    reg = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", XGBClassifier(random_state=42, eval_metric="logloss"))
    ])
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, 10],
        "model__min_child_weight": [1, 5, 10],
        "model__colsample_bytree": [0.5, 0.8, 1.0],
    }
    return grid_search(reg, param_grid, x_train, y_train)
