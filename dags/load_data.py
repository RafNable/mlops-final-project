import pandas as pd

# Define dataset path
dataset_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

# Display basic info
print("✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
print("\n📌 First 5 Rows:")
print(df.head())

# Check for missing values
print("\n🔍 Missing Values:")
print(df.isnull().sum())
