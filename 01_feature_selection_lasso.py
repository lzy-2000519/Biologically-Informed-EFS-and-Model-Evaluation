"""
LASSO Feature Selection with Bootstrap Resampling
Performs 100 iterations of bootstrap + 5-fold CV to identify stable features
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.utils import resample
from sklearn.model_selection import KFold
from tqdm import tqdm
import os

# ============================================================================
# Configuration - Modify paths according to your data location
# ============================================================================
DATA_DIR = "./data"  # Data directory (relative path)
RESULTS_DIR = "./results"  # Output directory for results
os.makedirs(RESULTS_DIR, exist_ok=True)

# **1. Data Import**
# Note: Place your data files in the 'data' folder or modify paths accordingly
X_filepath = f"{DATA_DIR}/metabolites_features.xlsx"
Y_filepath = f"{DATA_DIR}/metabolites_features.xlsx"

X = pd.read_excel(X_filepath, sheet_name='Sheet1', header=0, index_col=0)
Y = pd.read_excel(Y_filepath, sheet_name='Sheet2', usecols='A:B', header=0, index_col=0).values.ravel()

print(f"Original number of features: {X.shape[1]}")

# **2. Missing Value Filtering**
missing_rate = X.isnull().mean()
X = X.loc[:, missing_rate < 0.5]
print(f"Features remaining after filtering: {X.shape[1]}")

# **3. Set LASSO Training Iterations (100 repeats)**
n_iterations = 100
feature_counts = pd.Series(0, index=X.columns, dtype=float)
coef_matrix = np.zeros((n_iterations, X.shape[1]))

# Outer loop: 100 bootstrap iterations
for i in tqdm(range(n_iterations), desc="Bootstrap + 5-Fold CV"):
    # **Bootstrap Sampling (80% of data)**
    X_resampled, Y_resampled = resample(X, Y, n_samples=int(0.8 * len(Y)), random_state=i)

    # **Define 5-Fold Cross-Validation** (shuffle each iteration, random_state=i)
    kf = KFold(n_splits=5, shuffle=True, random_state=i)

    # **LassoCV with 5-Fold CV, auto-selects optimal alpha**
    lasso = LassoCV(alphas=np.logspace(-4, 1, 50), cv=kf, max_iter=10000)
    lasso.fit(X_resampled, Y_resampled)

    # **Record coefficients**
    coef_matrix[i, :] = lasso.coef_

    # **Count selected features (coefficient != 0)**
    selected_features = X.columns[lasso.coef_ != 0]
    feature_counts[selected_features] += 1

# **4. Calculate Feature Selection Frequency**
feature_freq = feature_counts / n_iterations
mean_coef = np.mean(coef_matrix, axis=0)      # Original coefficients (with sign)
mean_coef_abs = np.mean(np.abs(coef_matrix), axis=0)  # Absolute value mean
std_coef = np.std(coef_matrix, axis=0)

# **5. Generate DataFrame, sorted by absolute value but preserving sign**
feature_freq_df = pd.DataFrame({
    'Feature': X.columns,
    'Selection Frequency': feature_freq.values,
    'Mean Coefficient': mean_coef,         # Original coefficient with sign
    'Abs Mean Coefficient': mean_coef_abs, # Absolute value for sorting
    'Coefficient Std': std_coef
})

# Sort by absolute value in descending order
feature_freq_df = feature_freq_df.sort_values(by="Abs Mean Coefficient", ascending=False)

# **6. Save all feature selection frequencies**
output_path_freq = f"{RESULTS_DIR}/lasso_feature_selection_frequency.xlsx"
feature_freq_df.to_excel(output_path_freq, index=False)

# **7. Set Selection Threshold (example: frequency > 0.3)**
final_features = feature_freq_df[feature_freq_df['Selection Frequency'] > 0.3]['Feature']

# **8. Save Final Selected Features**
X_selected = X[final_features]
output_path_selected = f"{RESULTS_DIR}/lasso_stable_features.xlsx"
X_selected.to_excel(output_path_selected, index=True)

print(f"Number of stable features selected: {len(final_features)}, saved to {output_path_selected}")
print(f"All feature selection frequencies saved to {output_path_freq}")
