"""
Ensemble Feature Selection using Random Forest and XGBoost
With Bayesian Optimization and Permutation Importance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV
from sklearn.inspection import permutation_importance
from tqdm import tqdm

plt.rcParams['font.family'] = ['arial']

# ============================================================================
# Configuration - Modify paths according to your data location
# ============================================================================
DATA_DIR = "./data"  # Data directory (relative path)
OUTPUT_DIR = "./results"  # Output directory for results

# **1. Data Import**
# Note: Place your data files in the 'data' folder or modify paths accordingly
X_filepath = f"{DATA_DIR}/metabolites_features.xlsx"
Y_filepath = f"{DATA_DIR}/metabolites_features.xlsx"

X = pd.read_excel(X_filepath, sheet_name='Sheet1', header=0, index_col=0)
Y = pd.read_excel(Y_filepath, sheet_name='Sheet2', usecols='A:B', header=0, index_col=0).values.ravel()

# **2. Define Cross-Validation Strategy**
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# **3. Store Feature Importance and R² Scores**
rf_importance = pd.DataFrame(0, index=X.columns, columns=range(cv.get_n_splits()))
xgb_importance = pd.DataFrame(0, index=X.columns, columns=range(cv.get_n_splits()))

rf_r2_scores = []  # Store Random Forest R²
xgb_r2_scores = []  # Store XGBoost R²

# **4. Cross-Validation Loop**
for fold_idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X, Y)), total=cv.get_n_splits()):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # **4.1 Random Forest with Bayesian Optimization**
    rf = RandomForestRegressor(random_state=42)
    rf_search_space = {
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 20)
    }
    rf_bayes = BayesSearchCV(rf, rf_search_space, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    rf_bayes.fit(X_train, Y_train)
    best_rf = rf_bayes.best_estimator_

    # Calculate and record R²
    rf_r2_scores.append(best_rf.score(X_test, Y_test))

    # **Calculate Permutation Importance for each feature**
    result = permutation_importance(best_rf, X_test, Y_test, n_repeats=10, random_state=42)
    rf_importance.iloc[:, fold_idx] = result.importances_mean  # Store mean feature importance

    # **4.2 XGBoost with Bayesian Optimization**
    xgb = XGBRegressor(random_state=42)
    xgb_search_space = {
        'n_estimators': (50, 200),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.1),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.0, 1.0)
    }
    xgb_bayes = BayesSearchCV(xgb, xgb_search_space, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    xgb_bayes.fit(X_train, Y_train)
    best_xgb = xgb_bayes.best_estimator_

    # Calculate and record R²
    xgb_r2_scores.append(best_xgb.score(X_test, Y_test))

    # **Calculate Permutation Importance for each feature**
    result = permutation_importance(best_xgb, X_test, Y_test, n_repeats=10, random_state=42)
    xgb_importance.iloc[:, fold_idx] = result.importances_mean  # Store mean feature importance

# **5. Calculate Mean Feature Importance**
rf_mean_importance = rf_importance.mean(axis=1).sort_values(ascending=False)
xgb_mean_importance = xgb_importance.mean(axis=1).sort_values(ascending=False)

# **6. Save Feature Importance Results for Each Model**
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

rf_output_filepath = f"{OUTPUT_DIR}/RandomForest_Selected_Features.xlsx"
xgb_output_filepath = f"{OUTPUT_DIR}/XGBoost_Selected_Features.xlsx"

rf_mean_importance.to_excel(rf_output_filepath, header=['RF Permutation Importance'])
xgb_mean_importance.to_excel(xgb_output_filepath, header=['XGBoost Permutation Importance'])

# **7. Save R² Scores to Files**
rf_r2_filepath = f"{OUTPUT_DIR}/RandomForest_R2_Scores.xlsx"
xgb_r2_filepath = f"{OUTPUT_DIR}/XGBoost_R2_Scores.xlsx"

rf_r2_df = pd.DataFrame(rf_r2_scores, columns=['R²'])
xgb_r2_df = pd.DataFrame(xgb_r2_scores, columns=['R²'])

rf_r2_df.to_excel(rf_r2_filepath, index=False)
xgb_r2_df.to_excel(xgb_r2_filepath, index=False)

print("Feature importance and R² scores saved successfully.")
