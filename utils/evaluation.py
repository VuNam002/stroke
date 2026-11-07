from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    score = None
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_test)

    roc_auc = pr_auc = None
    if score is not None:
        roc_auc = roc_auc_score(y_test, score)
        pr_auc = average_precision_score(y_test, score)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        # Chỉ số của lớp dương tính (stroke = 1)
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        # Trung bình macro trên hai lớp
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }