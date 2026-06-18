import pandas as pd
import numpy as np

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
test_path = "/Users/admin/Documents/dev_src/stress_index/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

base_features = [
    "gender", "age", "height", "weight", "cholesterol", 
    "systolic_blood_pressure", "diastolic_blood_pressure", 
    "glucose", "bone_density", "activity", "smoke_status", 
    "medical_history", "family_medical_history", "sleep_pattern", 
    "edu_level", "mean_working"
]

base_features_num = [
    "age", "height", "weight", "cholesterol", 
    "systolic_blood_pressure", "diastolic_blood_pressure", 
    "glucose", "bone_density"
]

print("=== Exact Duplicates (All Features except ID) ===")
train_feats = train[base_features]
test_feats = test[base_features]

print("Train internal duplicates:", train_feats.duplicated().sum())
print("Test internal duplicates:", test_feats.duplicated().sum())

# Duplicates between Train and Test
merged = pd.merge(test.reset_index(), train, on=base_features, suffixes=('_test', '_train'))
print(f"Number of test rows that have an exact match in train (all 16 features): {merged['index'].nunique()} / {len(test)}")

print("\n=== Exact Duplicates (8 Numerical Features) ===")
train_feats_num = train[base_features_num]
test_feats_num = test[base_features_num]

print("Train internal duplicates (numerical):", train_feats_num.duplicated().sum())
print("Test internal duplicates (numerical):", test_feats_num.duplicated().sum())

merged_num = pd.merge(test.reset_index(), train, on=base_features_num, suffixes=('_test', '_train'))
print(f"Number of test rows that have an exact match in train (8 numerical features): {merged_num['index'].nunique()} / {len(test)}")

if len(merged_num) > 0:
    print("\nExample matches (8 numerical features):")
    print(merged_num[['index', 'ID_test', 'ID_train', 'stress_score']].head(10))
