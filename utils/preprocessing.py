from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

target = "stroke"
num_feature = ["age", "avg_glucose_level", "bmi"]
gender_value = ["gender"]
binary_feature = ["hypertension", "heart_disease", "urban_residence", "work_related_stress", "smokes"]
drop_feature = ["pat_id"]

def split_data(data, target = target, test_size = 0.2, random_state = 42):
    x = data.drop(columns=[target] + drop_feature)
    y = data[target]
    return train_test_split(x, y, test_size = test_size, random_state = random_state, stratify = y)

def build_preprocessing_pipeline():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    one_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    bool_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_feature),
        ("one", one_pipline, gender_value),
        ("bool", bool_pipline, binary_feature)
    ])

    return preprocessor