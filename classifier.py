from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "stroke_classification.csv"


def main():
    data = pd.read_csv(DATA_PATH)
    print(data.head())
    print(data.info())
    print(data["gender"].unique())

if __name__ == "__main__":
    main()