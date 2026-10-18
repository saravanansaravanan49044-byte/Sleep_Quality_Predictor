
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dataset location
DATA_PATH = "dataset/Sleep_health_and_lifestyle_dataset.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Check the CSV filename inside the dataset folder."
    )

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check required target column
target = "Quality of Sleep"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found. "
        "Check your dataset columns."
    )

# Convert sleep quality scores into categories
def classify_sleep(score):
    if score >= 7:
        return "Good"
    elif score >= 5:
        return "Average"
    return "Poor"

df["Sleep_Category"] = df[target].apply(classify_sleep)

# Use available lifestyle features
candidate_features = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Heart Rate",
    "Daily Steps",
    "Blood Pressure",
]

features = [
    column for column in candidate_features
    if column in df.columns
]

if not features:
    raise ValueError("No expected input features were found.")

X = df[features].copy()
y = df["Sleep_Category"]

# Fill missing values
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]

for column in numeric_features:
    X[column] = X[column].fillna(X[column].median())

for column in categorical_features:
    X[column] = X[column].fillna("Unknown")

# Prepare preprocessing and model
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numeric", "passthrough", numeric_features),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )),
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nModel training completed!")
print("Accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")
print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# Save the trained model and its expected input columns
os.makedirs("models", exist_ok=True)

joblib.dump(
    {"model": model, "features": features},
    "models/sleep_model.pkl",
)

print("\nModel saved to models/sleep_model.pkl")
print("Features used:", features)
