import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn


def create_run_dir(base_dir):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir)/"runs"/run_id
    (run_dir/"model").mkdir(parents=True)

    return run_dir

def save_model(model, path):
    joblib.dump(model, path)

def _to_builtin(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_to_builtin)

def build_run_info(data_path, data, **params):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "n_rows": int(data.shape[0]),
        "n_cols": int(data.shape[1]),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "params": params,
    }