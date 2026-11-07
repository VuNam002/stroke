from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


def grid_search(pipline, param_grid, x_train, y_train, cv=5, scoring="f1", n_jobs=-1, verbose=0):
    """Chạy GridSearchCV cho bài toán classifier, có thanh tiến trình.

    Trả về đối tượng GridSearchCV đã fit (dùng .best_estimator_, .best_params_, .best_score_).
    """
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
    )

    total = len(ParameterGrid(param_grid)) * cv.get_n_splits()

    with tqdm_joblib(tqdm(desc="GridSearchCV", total=total)):
        search.fit(x_train, y_train)

    print(f"Best {scoring}: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search
