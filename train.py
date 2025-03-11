import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Set MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Default")

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Separate target and features
target = "Attrition"
X = df.drop(columns=[target])
y = df[target].map({"Yes": 1, "No": 0})  # Convert target to numeric

# Identify categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=cat_cols)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log parameters and metrics
with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")

print("✅ Training completed successfully!")
