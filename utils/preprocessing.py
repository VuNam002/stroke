from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

target = "stroke"
num_feature = ["age", "avg_glucose_level", "bmi"]
gender_value = ["gender"]
binary_feature = ["hypertension", "heart_disease", "urban_residence", "stroke", "work_related_stress"]
drop_feature = ["pat_id"]

def split_data(data, target = target, test_size = 0.2, random_state = 42):
    x = data.drop(columns=[target])
    y = data[target]
    return train_test_split(x, y, test_size = test_size, random_state = random_state)

def build_preprocessing_pipeline():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
        ("scaler", StandardScaler())
    ])