import os

# Define the project structure
project_structure = [
    "data",  # Store raw and processed data
    "dags",  # Dagster pipeline scripts
    "models",  # Trained model artifacts
    "api",  # FastAPI service
    "mlflow_tracking",  # MLflow configuration
    "tests",  # Pytest unit tests
    "notebooks",  # Jupyter notebook demonstration
    "docker",  # Docker setup
]

# Create directories
for folder in project_structure:
    os.makedirs(folder, exist_ok=True)

# Create essential files
files = {
    "README.md": "# End-to-End ML System\n\nThis project implements an ML system for IBM HR Employee Attrition.",
    ".pre-commit-config.yaml": "repos:\n  - repo: https://github.com/charliermarsh/ruff\n    rev: v0.1.1\n    hooks:\n      - id: ruff",
    "docker-compose.yaml": "version: '3.8'\nservices:\n  fastapi:\n    build: ./api",
    "tests/__init__.py": "",
    "notebooks/exploration.ipynb": "",  # Placeholder for EDA notebook
}

# Create files with initial content
for filepath, content in files.items():
    with open(filepath, "w") as f:
        f.write(content)

print("✅ Project structure created successfully!")
