import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import pickle
import os

# Get path
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(_BASE_DIR, "Placement_Data_Expanded.csv")

# Load data
df = pd.read_csv(csv_path)

# Encoders dictionary
encoders = {}
categorical_cols = ['gender', 'degree', 'stream', 'workex']

# Copy dataframe for preprocessing
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le

# Store encoder classes for reference in Streamlit
encoder_classes = {col: list(le.classes_) for col, le in encoders.items()}

# Define Feature Columns
feature_cols = [
    'gender', 'degree', 'stream', 'ssc_p', 'hsc_p', 'cgpa', 'workex',
    'coding_skills', 'communication_skills', 'analytical_skills', 'domain_knowledge',
    'projects', 'internships', 'certifications'
]

# ----------------- 1. Placement Status Model -----------------
X_status = df_encoded[feature_cols]
y_status = df_encoded['placed_status'].apply(lambda x: 1 if x == 'Placed' else 0)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_status, y_status, test_size=0.2, random_state=42)

status_model = RandomForestClassifier(n_estimators=100, random_state=42)
status_model.fit(X_train_s, y_train_s)

y_pred_s = status_model.predict(X_test_s)
print("=== Placement Status Model ===")
print("Accuracy:", accuracy_score(y_test_s, y_pred_s))
print(classification_report(y_test_s, y_pred_s))

# ----------------- 2. Placed Sector Model -----------------
# Train only on placed records
df_placed = df_encoded[df_encoded['placed_status'] == 'Placed'].copy()

# Encode placed_sector
sector_encoder = LabelEncoder()
df_placed['placed_sector_encoded'] = sector_encoder.fit_transform(df_placed['placed_sector'])
encoders['placed_sector'] = sector_encoder

X_sector = df_placed[feature_cols]
y_sector = df_placed['placed_sector_encoded']

X_train_sec, X_test_sec, y_train_sec, y_test_sec = train_test_split(X_sector, y_sector, test_size=0.2, random_state=42)

sector_model = RandomForestClassifier(n_estimators=100, random_state=42)
sector_model.fit(X_train_sec, y_train_sec)

y_pred_sec = sector_model.predict(X_test_sec)
print("\n=== Placed Sector Model ===")
print("Accuracy:", accuracy_score(y_test_sec, y_pred_sec))
print(classification_report(y_test_sec, y_pred_sec, target_names=sector_encoder.classes_))

# ----------------- 3. Salary Predictor Model -----------------
# Train only on placed records
X_salary = df_placed[feature_cols]
y_salary = df_placed['salary_lpa']

X_train_sal, X_test_sal, y_train_sal, y_test_sal = train_test_split(X_salary, y_salary, test_size=0.2, random_state=42)

salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
salary_model.fit(X_train_sal, y_train_sal)

y_pred_sal = salary_model.predict(X_test_sal)
print("\n=== Salary Regressor Model ===")
print("Mean Absolute Error (LPA):", mean_absolute_error(y_test_sal, y_pred_sal))
print("R2 Score:", r2_score(y_test_sal, y_pred_sal))

# ----------------- Save Models & Encoders -----------------
models_dict = {
    'status_model': status_model,
    'sector_model': sector_model,
    'salary_model': salary_model,
    'encoders': encoders,
    'feature_cols': feature_cols,
    'encoder_classes': encoder_classes
}

model_save_path = os.path.join(_BASE_DIR, 'placement_models.pkl')
with open(model_save_path, 'wb') as f:
    pickle.dump(models_dict, f)

print(f"\nAll models and encoders successfully saved to: {model_save_path}")