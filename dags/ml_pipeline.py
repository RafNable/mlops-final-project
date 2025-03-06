import pandas as pd
import joblib
import os
from dagster import job, op, In, Out, Config, ConfigurableResource
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --- 1️⃣ Data Ingestion ---
@op(out=Out(pd.DataFrame))
def load_data():
    """Load the dataset from CSV."""
    dataset_path = "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(dataset_path)
    return df

# --- 2️⃣ Data Transformation ---
@op(ins={"df": In(pd.DataFrame)}, out=Out(pd.DataFrame))
def preprocess_data(df):
    """Preprocess data: handle missing values, encode categorical features, scale numerical features."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Scale numerical features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = StandardScaler().fit_transform(df[numerical_cols])

    return df

# --- 3️⃣ Model Training ---
@op(ins={"df": In(pd.DataFrame)}, out=Out(str))
def train_model(df):
    """Train a Random Forest model and save it as a file."""
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_path = "../models/random_forest_model.pkl"
    joblib.dump(model, model_path)
    return model_path

# --- 4️⃣ Model Validation ---
@op(ins={"model_path": In(str), "df": In(pd.DataFrame)}, out=Out(str))
def validate_model(model_path, df):
    """Validate model performance on test set."""
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"🎯 Model Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:\n", report)

    return f"Accuracy: {accuracy:.4f}"

# --- 5️⃣ Drift Detection ---
@op(ins={"df": In(pd.DataFrame)}, out=Out(str))
def detect_drift(df):
    """Generate a data drift report using Evidently AI."""
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=df.sample(frac=0.8), current_data=df.sample(frac=0.2))
    
    report_path = "../reports/data_drift_report.html"
    drift_report.save_html(report_path)
    return report_path

# --- 🚀 Define the Dagster Job ---
@job
def ml_pipeline():
    raw_data = load_data()
    processed_data = preprocess_data(raw_data)
    model_path = train_model(processed_data)
    validate_model(model_path, processed_data)
    detect_drift(processed_data)
