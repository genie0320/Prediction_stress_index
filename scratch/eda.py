import os
import pandas as pd
import numpy as np

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
test_path = "/Users/admin/Documents/dev_src/stress_index/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("=== Train Shape ===", train.shape)
print("=== Test Shape ===", test.shape)

print("\n=== Columns and Types ===")
for col in train.columns:
    null_count_tr = train[col].isnull().sum()
    null_count_te = test[col].isnull().sum() if col in test.columns else "-"
    unique_tr = train[col].nunique()
    unique_te = test[col].nunique() if col in test.columns else "-"
    print(f"{col:25} | Type: {str(train[col].dtype):7} | Train Nulls: {null_count_tr:4} (Uniq: {unique_tr:4}) | Test Nulls: {null_count_te:4} (Uniq: {unique_te:4})")

print("\n=== Target (stress_score) Distribution ===")
print(train['stress_score'].describe())
print("Number of unique target values:", train['stress_score'].nunique())
print("Top 10 target values:\n", train['stress_score'].value_counts().head(10))

print("\n=== Categorical Columns Exploration ===")
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    if col != 'ID':
        print(f"\n--- {col} ---")
        print("Train unique values:", train[col].unique())
        print("Test unique values:", test[col].unique())
        print("Train value counts:\n", train[col].value_counts(dropna=False).head(5))

print("\n=== Medical History / Family Medical History Examples ===")
print("Train medical_history (first 10):")
print(train['medical_history'].head(10))
print("\nTrain family_medical_history (first 10):")
print(train['family_medical_history'].head(10))

print("\n=== Sleep Pattern Examples ===")
print(train['sleep_pattern'].head(10))
