import pandas as pd
import joblib
import os
from dagster import job, op, In, Out, Config, ConfigurableResource
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable
from sklearn.preprocessing import OneHotEncoder

# --- 1️⃣ Data Ingestion ---
@op
def load_data():
    # Get the absolute path to ensure Dagster finds the file
    dataset_path = os.path.abspath("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    # Load the dataset
    df = pd.read_csv(dataset_path)
    return df

# --- 2️⃣ Data Transformation ---

@op(out={
    "X_train": Out(pd.DataFrame),
    "X_test": Out(pd.DataFrame),
    "y_train": Out(pd.Series),
    "y_test": Out(pd.Series),
})
def preprocess_data(df: pd.DataFrame):
    target_column = "Attrition"

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical variables using OneHotEncoder
    categorical_cols = X.select_dtypes(include=["object"]).columns
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
    X_encoded.columns = encoder.get_feature_names_out(categorical_cols)

    # Drop old categorical columns and replace them with encoded features
    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, X_encoded], axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test  # ✅ Return as a tuple


# --- 3️⃣ Model Training ---
@op(ins={"X_train": In(pd.DataFrame), "y_train": In(pd.Series)}, out=Out(str))
def train_model(X_train, y_train):
    # Define model save path
    model_dir = os.path.abspath("models")
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    os.makedirs(model_dir, exist_ok=True)

    # Train a simple Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_path)
    return model_path

# --- 4️⃣ Model Validation ---
@op(ins={"model_path": In(str), "X_test": In(pd.DataFrame), "y_test": In(pd.Series)}, out=Out(str))
def validate_model(model_path, X_test, y_test):
    """Validate model performance on test set."""
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
@op
def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    drift_report = Report(metrics=[DataDriftTable()])
    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    report_dir = os.path.abspath("reports")
    os.makedirs(report_dir, exist_ok=True)  # Ensure reports directory exists

    report_path = os.path.join(report_dir, "data_drift_report.html")
    drift_report.save_html(report_path)
    
    return report_path  # Ensure Dagster knows this step has output


# --- 🚀 Define the Dagster Job ---
@job
def ml_pipeline():
    raw_data = load_data()
    
    # Unpack outputs correctly
    X_train, X_test, y_train, y_test = preprocess_data(raw_data)

    model_path = train_model(X_train, y_train)
    validate_model(model_path, X_test, y_test)

    detect_drift(X_train, X_test)  # Ensure you're passing DataFrames
