"""
Five-Model Regression Analysis - Complete Version
Comprehensive analysis of 5 regression models:
- ElasticNet
- Bayesian Ridge  
- Linear Regression
- PLS
- SVR
Includes: Cook's distance outlier detection, hyperparameter optimization, 
5x100 CV, Bootstrap, Cross-seed validation
"""
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import (cross_val_score, train_test_split, KFold, 
                                     RepeatedKFold, GridSearchCV)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.stats import shapiro
import warnings
import time
import os
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration - Modify paths according to your data location
# ============================================================================
DATA_DIR = "./data"  # Data directory (relative path)
RESULTS_DIR = "./results"  # Output directory for results
MODELS_DIR = "./results/trained_models"  # Trained models directory

print("="*80)
print("FIVE-MODEL REGRESSION COMPARISON FOR METABOLOMICS")
print("="*80)

# ============================================================================
# Data Loading (Auto-detect feature columns)
# ============================================================================
# Note: Place your data files in the 'data' folder or modify paths accordingly
filepath = f"{DATA_DIR}/selected_features.xlsx"
features_df = pd.read_excel(filepath, sheet_name=0)
target_df = pd.read_excel(filepath, sheet_name=1)

# Auto-detect numeric feature columns from the first sheet (skip sample ID column)
numeric_features = features_df.columns[1:].tolist()  # Start from column 2

X = features_df[numeric_features].values
y = target_df.iloc[:, 1].values  # Read column 2 as target variable (column 1 is sample ID)

print(f"\n📊 Dataset: N = {len(X)}, Features = {len(numeric_features)}")
print(f"   Feature names: {numeric_features}")
print(f"   Target variable: {target_df.columns[1]}")

# ============================================================================
# Outlier Detection using Cook's Distance
# ============================================================================
print("\n" + "="*80)
print("OUTLIER DETECTION (Cook's Distance)")
print("="*80)

lr_temp = LinearRegression()
lr_temp.fit(X, y)
residuals_temp = y - lr_temp.predict(X)
std_residuals = residuals_temp / np.std(residuals_temp)

X_with_intercept = np.column_stack([np.ones(len(X)), X])
H = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
leverage = np.diag(H)

n, p = X.shape
cooks_d = (std_residuals**2 / p) * (leverage / (1 - leverage)**2)
cook_threshold = 4 / n
outliers_mask = cooks_d > cook_threshold
n_outliers = outliers_mask.sum()

print(f"\n🔍 Outlier Analysis:")
print(f"   Threshold: {cook_threshold:.4f}")
print(f"   Outliers: {n_outliers}/{len(X)} ({n_outliers/len(X)*100:.1f}%)")

if n_outliers > 0 and n_outliers / len(X) < 0.1:
    X_clean = X[~outliers_mask]
    y_clean = y[~outliers_mask]
    print(f"   ✓ Using cleaned dataset: N = {len(X_clean)}")
    X_use, y_use = X_clean, y_clean
else:
    X_use, y_use = X, y
    print(f"   ✓ Using original dataset")

# ============================================================================
# Train-Test Split (consistent with all models, random_state=42)
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_use, y_use, test_size=0.3, random_state=42, shuffle=True
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# ============================================================================
# Define Models and Hyperparameter Grids
# ============================================================================
print("\n" + "="*80)
print("MODEL CONFIGURATION")
print("="*80)

models_config = {
    'ElasticNet': {
        'model': ElasticNet(max_iter=50000, random_state=42),
        'param_grid': {
            'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        },
        'needs_tuning': True
    },
    'Bayesian Ridge': {
        'model': BayesianRidge(max_iter=2000, compute_score=True),
        'param_grid': None,
        'needs_tuning': False
    },
    'Linear Regression': {
        'model': LinearRegression(),
        'param_grid': None,
        'needs_tuning': False
    },
    'PLS': {
        'model': PLSRegression(scale=False),  # Data already standardized
        'param_grid': {
            'n_components': list(range(1, min(11, len(X_train))))
        },
        'needs_tuning': True
    },
    'SVR': {
        'model': SVR(),
        'param_grid': {
            'C': [0.1, 1, 5, 10, 20, 50, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'epsilon': [0.01, 0.05, 0.1, 0.2]
        },
        'needs_tuning': True
    }
}

print("\n✓ Configured 5 models:")
for model_name in models_config.keys():
    print(f"   • {model_name}")

# ============================================================================
# Train and Evaluate All Models
# ============================================================================
all_results = []

for model_name, config in models_config.items():
    print("\n" + "="*80)
    print(f"PROCESSING: {model_name}")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Hyperparameter Optimization
    if config['needs_tuning'] and config['param_grid']:
        print(f"\n⏳ Tuning hyperparameters...")
        inner_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        grid = GridSearchCV(config['model'], config['param_grid'], 
                           cv=inner_cv, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        model_to_use = grid.best_estimator_
        best_params = grid.best_params_
        print(f"   ✓ Best params: {best_params}")
    else:
        model_to_use = config['model']
        model_to_use.fit(X_train, y_train)
        best_params = None
        print(f"   ✓ No tuning needed")
    
    # 2. Train and Test Set Predictions
    y_train_pred = model_to_use.predict(X_train)
    y_test_pred = model_to_use.predict(X_test)
    
    # PLS predictions need to be flattened
    if model_name == 'PLS':
        y_train_pred = y_train_pred.ravel()
        y_test_pred = y_test_pred.ravel()
    
    # 3. Basic Performance Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Adjusted R²
    n_test = len(y_test)
    n_params = len(best_params) if best_params else 1
    test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - n_params - 1)
    
    # Overfitting analysis
    overfit = train_r2 - test_r2
    
    print(f"\n📊 Basic Performance:")
    print(f"   Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")
    print(f"   Overfit Gap = {overfit:+.4f}")
    print(f"   Test RMSE = {test_rmse:.4f}, Test MAE = {test_mae:.4f}")
     
    # 4. Residual Analysis
    residuals_test = y_test - y_test_pred
    shapiro_stat, p_shapiro = shapiro(residuals_test)
    
    # 5. Repeated Cross-Validation (5×100)
    print(f"\n⏳ Running 5×100 Repeated CV...")
    repeated_cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)
    # Use cross_validate to get train and validation scores (for calculating mean/STD/SEM)
    from sklearn.model_selection import cross_validate
    cv_res = cross_validate(
        model_to_use, X_use, y_use,
        cv=repeated_cv,
        scoring=('r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'),
        return_train_score=True,
        n_jobs=-1
    )

    # Extract validation (test) scores
    cv_test_r2 = cv_res['test_r2']
    cv_test_rmse = -cv_res['test_neg_root_mean_squared_error']
    cv_test_mae = -cv_res['test_neg_mean_absolute_error']

    # Extract training scores
    cv_train_r2 = cv_res['train_r2']
    cv_train_rmse = -cv_res['train_neg_root_mean_squared_error']
    cv_train_mae = -cv_res['train_neg_mean_absolute_error']

    # Calculate statistics (validation/test)
    cv_mean = np.mean(cv_test_r2)
    cv_std = np.std(cv_test_r2)
    cv_sem = cv_std / np.sqrt(len(cv_test_r2))
    cv_ci = [np.percentile(cv_test_r2, 2.5), np.percentile(cv_test_r2, 97.5)]
    cv_rmse_mean = np.mean(cv_test_rmse)
    cv_rmse_std = np.std(cv_test_rmse)
    cv_rmse_sem = cv_rmse_std / np.sqrt(len(cv_test_rmse))
    cv_mae_mean = np.mean(cv_test_mae)
    cv_mae_std = np.std(cv_test_mae)
    cv_mae_sem = cv_mae_std / np.sqrt(len(cv_test_mae))

    # Calculate statistics (training set)
    cv_train_mean = np.mean(cv_train_r2)
    cv_train_std = np.std(cv_train_r2)
    cv_train_sem = cv_train_std / np.sqrt(len(cv_train_r2))
    cv_train_rmse_mean = np.mean(cv_train_rmse)
    cv_train_rmse_std = np.std(cv_train_rmse)
    cv_train_rmse_sem = cv_train_rmse_std / np.sqrt(len(cv_train_rmse))
    cv_train_mae_mean = np.mean(cv_train_mae)
    cv_train_mae_std = np.std(cv_train_mae)
    cv_train_mae_sem = cv_train_mae_std / np.sqrt(len(cv_train_mae))

    print(f"   CV (test) R² = {cv_mean:.4f} ± {cv_std:.4f} (SEM={cv_sem:.4f})")
    print(f"   CV (train) R² = {cv_train_mean:.4f} ± {cv_train_std:.4f} (SEM={cv_train_sem:.4f})")
    print(f"   CV RMSE (test) = {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f} (SEM={cv_rmse_sem:.4f})")
    print(f"   CV RMSE (train) = {cv_train_rmse_mean:.4f} ± {cv_train_rmse_std:.4f} (SEM={cv_train_rmse_sem:.4f})")
    print(f"   CV MAE (test) = {cv_mae_mean:.4f} ± {cv_mae_std:.4f} (SEM={cv_mae_sem:.4f})")
    print(f"   CV MAE (train) = {cv_train_mae_mean:.4f} ± {cv_train_mae_std:.4f} (SEM={cv_train_mae_sem:.4f})")
    print(f"   CV Stability = {cv_std/cv_mean*100:.2f}%")
    
    # 6. Bootstrap Validation
    print(f"\n⏳ Running Bootstrap (n=1000)...")
    n_bootstrap = 1000
    bootstrap_r2 = []
    bootstrap_rmse = []
    bootstrap_mae = []
    
    np.random.seed(42)
    for i in range(n_bootstrap):
        indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_boot = X_test[indices]
        y_boot = y_test[indices]
        
        y_boot_pred = model_to_use.predict(X_boot)
        if model_name == 'PLS':
            y_boot_pred = y_boot_pred.ravel()
        
        try:
            r2_boot = r2_score(y_boot, y_boot_pred)
            rmse_boot = np.sqrt(mean_squared_error(y_boot, y_boot_pred))
            mae_boot = mean_absolute_error(y_boot, y_boot_pred)
            bootstrap_r2.append(r2_boot)
            bootstrap_rmse.append(rmse_boot)
            bootstrap_mae.append(mae_boot)
        except:
            continue
    
    bootstrap_r2 = np.array(bootstrap_r2)
    bootstrap_rmse = np.array(bootstrap_rmse)
    bootstrap_mae = np.array(bootstrap_mae)
    # Also calculate bootstrap statistics on training set (with replacement sampling)
    bootstrap_train_r2 = []
    bootstrap_train_rmse = []
    bootstrap_train_mae = []
    for i in range(n_bootstrap):
        idx_tr = np.random.choice(len(X_train), size=len(X_train), replace=True)
        Xb_tr = X_train[idx_tr]
        yb_tr = y_train[idx_tr]
        yb_tr_pred = model_to_use.predict(Xb_tr)
        if model_name == 'PLS':
            yb_tr_pred = yb_tr_pred.ravel()
        try:
            bootstrap_train_r2.append(r2_score(yb_tr, yb_tr_pred))
            bootstrap_train_rmse.append(np.sqrt(mean_squared_error(yb_tr, yb_tr_pred)))
            bootstrap_train_mae.append(mean_absolute_error(yb_tr, yb_tr_pred))
        except:
            continue
    bootstrap_train_r2 = np.array(bootstrap_train_r2)
    bootstrap_train_rmse = np.array(bootstrap_train_rmse)
    bootstrap_train_mae = np.array(bootstrap_train_mae)
    
    bootstrap_mean = bootstrap_r2.mean()
    bootstrap_std = bootstrap_r2.std()
    bootstrap_sem = bootstrap_std / np.sqrt(len(bootstrap_r2)) if len(bootstrap_r2) > 0 else np.nan
    bootstrap_ci = [np.percentile(bootstrap_r2, 2.5), np.percentile(bootstrap_r2, 97.5)]
    bootstrap_rmse_mean = bootstrap_rmse.mean()
    bootstrap_rmse_std = bootstrap_rmse.std()
    bootstrap_rmse_sem = bootstrap_rmse_std / np.sqrt(len(bootstrap_rmse)) if len(bootstrap_rmse) > 0 else np.nan
    bootstrap_mae_mean = bootstrap_mae.mean()
    bootstrap_mae_std = bootstrap_mae.std()
    bootstrap_mae_sem = bootstrap_mae_std / np.sqrt(len(bootstrap_mae)) if len(bootstrap_mae) > 0 else np.nan

    # Training set bootstrap statistics
    bootstrap_train_mean = bootstrap_train_r2.mean() if len(bootstrap_train_r2) > 0 else np.nan
    bootstrap_train_std = bootstrap_train_r2.std() if len(bootstrap_train_r2) > 0 else np.nan
    bootstrap_train_sem = bootstrap_train_std / np.sqrt(len(bootstrap_train_r2)) if len(bootstrap_train_r2) > 0 else np.nan
    bootstrap_train_ci = [np.percentile(bootstrap_train_r2, 2.5), np.percentile(bootstrap_train_r2, 97.5)] if len(bootstrap_train_r2) > 0 else [np.nan, np.nan]
    bootstrap_train_rmse_mean = bootstrap_train_rmse.mean() if len(bootstrap_train_rmse) > 0 else np.nan
    bootstrap_train_rmse_std = bootstrap_train_rmse.std() if len(bootstrap_train_rmse) > 0 else np.nan
    bootstrap_train_rmse_sem = bootstrap_train_rmse_std / np.sqrt(len(bootstrap_train_rmse)) if len(bootstrap_train_rmse) > 0 else np.nan
    bootstrap_train_mae_mean = bootstrap_train_mae.mean() if len(bootstrap_train_mae) > 0 else np.nan
    bootstrap_train_mae_std = bootstrap_train_mae.std() if len(bootstrap_train_mae) > 0 else np.nan
    bootstrap_train_mae_sem = bootstrap_train_mae_std / np.sqrt(len(bootstrap_train_mae)) if len(bootstrap_train_mae) > 0 else np.nan
    
    print(f"   Bootstrap R² = {bootstrap_mean:.4f} ± {bootstrap_std:.4f} (SEM={bootstrap_sem:.4f})")
    print(f"   Bootstrap RMSE = {bootstrap_rmse_mean:.4f} ± {bootstrap_rmse_std:.4f} (SEM={bootstrap_rmse_sem:.4f}), MAE = {bootstrap_mae_mean:.4f} ± {bootstrap_mae_std:.4f} (SEM={bootstrap_mae_sem:.4f})")
    print(f"   Bootstrap CI = [{bootstrap_ci[0]:.3f}, {bootstrap_ci[1]:.3f}]")
    
    # 7. Cross-seed Validation
    print(f"\n⏳ Running Cross-seed validation (10 seeds)...")
    seeds = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
    seed_results = []
    
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_use, y_use, test_size=0.3, random_state=seed, shuffle=True
        )
        
        # Retrain model
        if config['needs_tuning'] and config['param_grid'] and best_params:
            model_temp = config['model'].__class__(**best_params)
        else:
            model_temp = config['model'].__class__()
        
        model_temp.fit(X_tr, y_tr)
        y_pred = model_temp.predict(X_te)
        
        if model_name == 'PLS':
            y_pred = y_pred.ravel()
        
        r2 = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae = mean_absolute_error(y_te, y_pred)
        # Also record training set metrics
        y_tr_pred = model_temp.predict(X_tr)
        if model_name == 'PLS':
            y_tr_pred = y_tr_pred.ravel()
        tr_r2 = r2_score(y_tr, y_tr_pred)
        tr_rmse = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
        tr_mae = mean_absolute_error(y_tr, y_tr_pred)
        seed_results.append({'seed': seed, 'test_r2': r2, 'test_rmse': rmse, 'test_mae': mae,
                             'train_r2': tr_r2, 'train_rmse': tr_rmse, 'train_mae': tr_mae})
    
    seed_df = pd.DataFrame(seed_results)
    seed_mean = seed_df['test_r2'].mean()
    seed_std = seed_df['test_r2'].std()
    seed_sem = seed_std / np.sqrt(len(seed_df))
    seed_train_mean = seed_df['train_r2'].mean()
    seed_train_std = seed_df['train_r2'].std()
    seed_train_sem = seed_train_std / np.sqrt(len(seed_df))
    seed_ci = [seed_df['test_r2'].quantile(0.025), seed_df['test_r2'].quantile(0.975)]
    seed_rmse_mean = seed_df['test_rmse'].mean()
    seed_rmse_std = seed_df['test_rmse'].std()
    seed_rmse_sem = seed_rmse_std / np.sqrt(len(seed_df))
    seed_mae_mean = seed_df['test_mae'].mean()
    seed_mae_std = seed_df['test_mae'].std()
    seed_mae_sem = seed_mae_std / np.sqrt(len(seed_df))
    
    print(f"   Cross-seed R² (test) = {seed_mean:.4f} ± {seed_std:.4f} (SEM={seed_sem:.4f})")
    print(f"   Cross-seed R² (train) = {seed_train_mean:.4f} ± {seed_train_std:.4f} (SEM={seed_train_sem:.4f})")
    print(f"   Cross-seed RMSE (test) = {seed_rmse_mean:.4f} ± {seed_rmse_std:.4f} (SEM={seed_rmse_sem:.4f}), MAE = {seed_mae_mean:.4f} ± {seed_mae_std:.4f} (SEM={seed_mae_sem:.4f})")
    print(f"   Cross-seed RMSE (train) = {seed_df['train_rmse'].mean():.4f} ± {seed_df['train_rmse'].std():.4f} (SEM={seed_df['train_rmse'].std()/np.sqrt(len(seed_df)):.4f})")
    print(f"   Cross-seed MAE (train) = {seed_df['train_mae'].mean():.4f} ± {seed_df['train_mae'].std():.4f} (SEM={seed_df['train_mae'].std()/np.sqrt(len(seed_df)):.4f})")
    print(f"   Cross-seed Stability = {seed_std/seed_mean*100:.2f}%")
    
    # 8. Statistical Testing
    if test_r2 > 0:
        f_stat = (test_r2 / (1 - test_r2)) * ((n_test - n_params - 1) / n_params)
        p_value = 1 - stats.f.cdf(f_stat, n_params, n_test - n_params - 1)
    else:
        f_stat = np.nan
        p_value = np.nan
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Completed in {elapsed_time:.1f}s")
    
    # 9. Save Results
    all_results.append({
        'Model': model_name,
        'Best_Params': str(best_params) if best_params else 'None',
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Adjusted_R2': test_adj_r2,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Overfit_Gap': overfit,
        'Shapiro_p': p_shapiro,
        'F_stat': f_stat,
        'F_p_value': p_value,
        # CV (train-side)
        'CV_Train_R2_Mean': cv_train_mean,
        'CV_Train_R2_Std': cv_train_std,
        'CV_Train_R2_SEM': cv_train_sem,
        'CV_Train_RMSE_Mean': cv_train_rmse_mean,
        'CV_Train_RMSE_Std': cv_train_rmse_std,
        'CV_Train_RMSE_SEM': cv_train_rmse_sem,
        'CV_Train_MAE_Mean': cv_train_mae_mean,
        'CV_Train_MAE_Std': cv_train_mae_std,
        'CV_Train_MAE_SEM': cv_train_mae_sem,
        'CV_R2_Mean': cv_mean,
        'CV_R2_Std': cv_std,
        'CV_R2_SEM': cv_sem,
        'CV_CI_Lower': cv_ci[0],
        'CV_CI_Upper': cv_ci[1],
        'CV_Stability_Pct': cv_std/cv_mean*100,
        'CV_RMSE_Mean': cv_rmse_mean,
        'CV_RMSE_Std': cv_rmse_std,
        'CV_RMSE_SEM': cv_rmse_sem,
        'CV_MAE_Mean': cv_mae_mean,
        'CV_MAE_Std': cv_mae_std,
        'CV_MAE_SEM': cv_mae_sem,
        'Bootstrap_R2_Mean': bootstrap_mean,
        'Bootstrap_R2_Std': bootstrap_std,
        'Bootstrap_R2_SEM': bootstrap_sem,
        'Bootstrap_CI_Lower': bootstrap_ci[0],
        'Bootstrap_CI_Upper': bootstrap_ci[1],
        'Bootstrap_RMSE_Mean': bootstrap_rmse_mean,
        'Bootstrap_RMSE_Std': bootstrap_rmse_std,
        'Bootstrap_RMSE_SEM': bootstrap_rmse_sem,
        'Bootstrap_MAE_Mean': bootstrap_mae_mean,
        'Bootstrap_MAE_Std': bootstrap_mae_std,
        'Bootstrap_MAE_SEM': bootstrap_mae_sem,
        # Bootstrap (train-side)
        'Bootstrap_Train_R2_Mean': bootstrap_train_mean,
        'Bootstrap_Train_R2_Std': bootstrap_train_std,
        'Bootstrap_Train_R2_SEM': bootstrap_train_sem,
        'Bootstrap_Train_CI_Lower': bootstrap_train_ci[0],
        'Bootstrap_Train_CI_Upper': bootstrap_train_ci[1],
        'Bootstrap_Train_RMSE_Mean': bootstrap_train_rmse_mean,
        'Bootstrap_Train_RMSE_Std': bootstrap_train_rmse_std,
        'Bootstrap_Train_RMSE_SEM': bootstrap_train_rmse_sem,
        'Bootstrap_Train_MAE_Mean': bootstrap_train_mae_mean,
        'Bootstrap_Train_MAE_Std': bootstrap_train_mae_std,
        'Bootstrap_Train_MAE_SEM': bootstrap_train_mae_sem,
        'CrossSeed_R2_Mean': seed_mean,
        'CrossSeed_R2_Std': seed_std,
        'CrossSeed_R2_SEM': seed_sem,
        'CrossSeed_CI_Lower': seed_ci[0],
        'CrossSeed_CI_Upper': seed_ci[1],
        'CrossSeed_Stability_Pct': seed_std/seed_mean*100,
        'CrossSeed_RMSE_Mean': seed_rmse_mean,
        'CrossSeed_RMSE_Std': seed_rmse_std,
        'CrossSeed_RMSE_SEM': seed_rmse_sem,
        'CrossSeed_MAE_Mean': seed_mae_mean,
        'CrossSeed_MAE_Std': seed_mae_std,
        'CrossSeed_MAE_SEM': seed_mae_sem,
        # Cross-seed (train-side)
        'CrossSeed_Train_R2_Mean': seed_train_mean,
        'CrossSeed_Train_R2_Std': seed_train_std,
        'CrossSeed_Train_R2_SEM': seed_train_sem,
        'CrossSeed_Train_RMSE_Mean': seed_df['train_rmse'].mean(),
        'CrossSeed_Train_RMSE_Std': seed_df['train_rmse'].std(),
        'CrossSeed_Train_RMSE_SEM': seed_df['train_rmse'].std()/np.sqrt(len(seed_df)) if len(seed_df)>0 else np.nan,
        'CrossSeed_Train_MAE_Mean': seed_df['train_mae'].mean(),
        'CrossSeed_Train_MAE_Std': seed_df['train_mae'].std(),
        'CrossSeed_Train_MAE_SEM': seed_df['train_mae'].std()/np.sqrt(len(seed_df)) if len(seed_df)>0 else np.nan,
        'CV_Scores': cv_test_r2,
        'Bootstrap_Scores': bootstrap_r2,
        'Seed_Results': seed_df,
        'Residuals': residuals_test,
        'y_test_pred': y_test_pred,
        'Training_Time_s': elapsed_time,
        'Model_Object': model_to_use
    })

# ============================================================================
# Results Summary
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame([{k: v for k, v in r.items() 
                            if k not in ['CV_Scores', 'Bootstrap_Scores', 
                                        'Seed_Results', 'Residuals', 
                                        'y_test_pred', 'Model_Object']} 
                          for r in all_results])

# Display main metrics
display_cols = ['Model', 'Test_R2', 'CV_R2_Mean', 'CrossSeed_R2_Mean', 
                'Bootstrap_R2_Mean', 'Overfit_Gap', 'CV_Stability_Pct']
print("\n" + results_df[display_cols].to_string(index=False))

# Ranking
print("\n" + "="*80)
print("MODEL RANKING (by CV R²)")
print("="*80)

results_sorted = results_df.sort_values('CV_R2_Mean', ascending=False)
for i, (idx, row) in enumerate(results_sorted.iterrows(), 1):
    medal = ['🥇', '🥈', '🥉', '4.', '5.'][i-1]
    print(f"\n{medal} {row['Model']:<18}")
    print(f"   CV R² = {row['CV_R2_Mean']:.4f} ± {row['CV_R2_Std']:.4f}")
    print(f"   Test R² = {row['Test_R2']:.4f}")
    print(f"   Cross-seed R² = {row['CrossSeed_R2_Mean']:.4f} (CV={row['CrossSeed_Stability_Pct']:.1f}%)")
    print(f"   Bootstrap R² = {row['Bootstrap_R2_Mean']:.4f}")
    print(f"   Overfit = {row['Overfit_Gap']:+.4f}")

# ============================================================================
# Save Detailed Results to Excel
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS TO EXCEL")
print("="*80)

os.makedirs(RESULTS_DIR, exist_ok=True)
output_excel = f"{RESULTS_DIR}/five_models_detailed_results.xlsx"

# Delete old Excel file (if exists)
if os.path.exists(output_excel):
    os.remove(output_excel)
    print(f"   ✓ Removed old Excel file")

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Sheet 1: Main Performance Metrics
    main_metrics = results_df[[
        'Model', 'Best_Params', 'Train_R2', 'Test_R2', 'Adjusted_R2',
        'Train_RMSE', 'Test_RMSE', 'Train_MAE', 'Test_MAE', 'Overfit_Gap',
        'Shapiro_p', 'F_stat', 'F_p_value'
    ]]
    main_metrics.to_excel(writer, sheet_name='Main_Metrics', index=False)
    
    # Sheet 2: Cross-Validation Results
    cv_metrics = results_df[[
        'Model',
        'CV_R2_Mean', 'CV_R2_Std', 'CV_R2_SEM', 'CV_CI_Lower', 'CV_CI_Upper',
        'CV_Stability_Pct',
        'CV_RMSE_Mean', 'CV_RMSE_Std', 'CV_RMSE_SEM',
        'CV_MAE_Mean', 'CV_MAE_Std', 'CV_MAE_SEM',
        # train-side
        'CV_Train_R2_Mean', 'CV_Train_R2_Std', 'CV_Train_R2_SEM',
        'CV_Train_RMSE_Mean', 'CV_Train_RMSE_Std', 'CV_Train_RMSE_SEM',
        'CV_Train_MAE_Mean', 'CV_Train_MAE_Std', 'CV_Train_MAE_SEM'
    ]]
    cv_metrics.to_excel(writer, sheet_name='CrossValidation', index=False)
    
    # Sheet 3: Bootstrap Results
    bootstrap_metrics = results_df[[
        'Model', 'Bootstrap_R2_Mean', 'Bootstrap_R2_Std', 'Bootstrap_R2_SEM',
        'Bootstrap_CI_Lower', 'Bootstrap_CI_Upper',
        'Bootstrap_RMSE_Mean', 'Bootstrap_RMSE_Std', 'Bootstrap_RMSE_SEM',
        'Bootstrap_MAE_Mean', 'Bootstrap_MAE_Std', 'Bootstrap_MAE_SEM',
        # train-side
        'Bootstrap_Train_R2_Mean', 'Bootstrap_Train_R2_Std', 'Bootstrap_Train_R2_SEM',
        'Bootstrap_Train_CI_Lower', 'Bootstrap_Train_CI_Upper',
        'Bootstrap_Train_RMSE_Mean', 'Bootstrap_Train_RMSE_Std', 'Bootstrap_Train_RMSE_SEM',
        'Bootstrap_Train_MAE_Mean', 'Bootstrap_Train_MAE_Std', 'Bootstrap_Train_MAE_SEM'
    ]]
    bootstrap_metrics.to_excel(writer, sheet_name='Bootstrap', index=False)
    
    # Sheet 4: Cross-seed Results
    seed_metrics = results_df[[
        'Model', 'CrossSeed_R2_Mean', 'CrossSeed_R2_Std', 'CrossSeed_R2_SEM',
        'CrossSeed_CI_Lower', 'CrossSeed_CI_Upper', 'CrossSeed_Stability_Pct',
        'CrossSeed_RMSE_Mean', 'CrossSeed_RMSE_Std', 'CrossSeed_RMSE_SEM',
        'CrossSeed_MAE_Mean', 'CrossSeed_MAE_Std', 'CrossSeed_MAE_SEM',
        # train-side
        'CrossSeed_Train_R2_Mean', 'CrossSeed_Train_R2_Std', 'CrossSeed_Train_R2_SEM',
        'CrossSeed_Train_RMSE_Mean', 'CrossSeed_Train_RMSE_Std', 'CrossSeed_Train_RMSE_SEM',
        'CrossSeed_Train_MAE_Mean', 'CrossSeed_Train_MAE_Std', 'CrossSeed_Train_MAE_SEM'
    ]]
    seed_metrics.to_excel(writer, sheet_name='CrossSeed', index=False)
    
    # Sheet 5: Overall Ranking
    ranking_data = []
    for i, (idx, row) in enumerate(results_sorted.iterrows(), 1):
        ranking_data.append({
            'Rank': i,
            'Model': row['Model'],
            'CV_R2': row['CV_R2_Mean'],
            'Test_R2': row['Test_R2'],
            'CrossSeed_R2': row['CrossSeed_R2_Mean'],
            'Bootstrap_R2': row['Bootstrap_R2_Mean'],
            'Overfit_Gap': row['Overfit_Gap'],
            'CrossSeed_Stability': f"{row['CrossSeed_Stability_Pct']:.2f}%"
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df.to_excel(writer, sheet_name='Ranking', index=False)
    
    # Sheet 6-10: Detailed CV Scores for Each Model
    for result in all_results:
        cv_detail = pd.DataFrame({
            'Fold': range(1, len(result['CV_Scores'])+1),
            'R2_Score': result['CV_Scores']
        })
        sheet_name = f"{result['Model']}_CV"[:31]  # Excel sheet name limit: 31 chars
        cv_detail.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✓ Results saved: {output_excel}")

# ============================================================================
# Final Report
# ============================================================================
print("\n" + "="*80)
print("FINAL REPORT")
print("="*80)

best_model = results_sorted.iloc[0]

print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"\n   Why it's best:")
print(f"   • Highest CV R² = {best_model['CV_R2_Mean']:.4f} ± {best_model['CV_R2_Std']:.4f}")
print(f"   • Test R² = {best_model['Test_R2']:.4f}")
print(f"   • Cross-seed R² = {best_model['CrossSeed_R2_Mean']:.4f}")
print(f"   • Cross-seed stability = {best_model['CrossSeed_Stability_Pct']:.1f}%")
print(f"   • Bootstrap R² = {best_model['Bootstrap_R2_Mean']:.4f}")
print(f"   • Overfit gap = {best_model['Overfit_Gap']:+.4f}")

print(f"\n📊 PERFORMANCE SUMMARY:")
print(f"\n   Model Rankings (by CV R²):")
for i, (idx, row) in enumerate(results_sorted.iterrows(), 1):
    medal = ['🥇', '🥈', '🥉', '  ', '  '][i-1]
    print(f"   {medal} {i}. {row['Model']:<18} CV R²={row['CV_R2_Mean']:.4f}")

print(f"\n📈 KEY INSIGHTS:")

# Find the most stable model
most_stable = results_df.loc[results_df['CrossSeed_Stability_Pct'].idxmin()]
print(f"   • Most stable: {most_stable['Model']} ({most_stable['CrossSeed_Stability_Pct']:.1f}% CV)")

# Find the model with least overfitting
least_overfit = results_df.loc[results_df['Overfit_Gap'].abs().idxmin()]
print(f"   • Least overfitting: {least_overfit['Model']} (Gap={least_overfit['Overfit_Gap']:+.4f})")

# Find the most accurate model (Test R²)
most_accurate = results_df.loc[results_df['Test_R2'].idxmax()]
print(f"   • Highest test accuracy: {most_accurate['Model']} (R²={most_accurate['Test_R2']:.4f})")

print(f"\n💡 RECOMMENDATION FOR PUBLICATION:")
print(f"""
   Primary Model: {best_model['Model']}
   - CV R² = {best_model['CV_R2_Mean']:.4f} ± {best_model['CV_R2_Std']:.4f}
   - Test R² = {best_model['Test_R2']:.4f}
   - RMSE = {best_model['Test_RMSE']:.4f}
   - MAE = {best_model['Test_MAE']:.4f}
   
   Validation:
   - 5×100 repeated cross-validation
   - Bootstrap validation (n=1000)
   - Cross-seed validation (10 seeds, {best_model['CrossSeed_Stability_Pct']:.1f}% CV)
   
   Dataset:
   - Samples: {len(X_use)} (outliers removed: {n_outliers})
   - Features: {len(numeric_features)} metabolites
   - Train/Test split: 70/30 (random_state=42)
""")

# ============================================================================
# Save Trained Models (for SHAP Analysis)
# ============================================================================

models_output_dir = MODELS_DIR
os.makedirs(models_output_dir, exist_ok=True)

for result in all_results:
    model_name = result['Model']
    model_obj = result['Model_Object']
    model_path = f"{models_output_dir}/{model_name}_model.pkl"
    joblib.dump(model_obj, model_path)
    print(f"   ✓ Saved model: {model_path}")

print("\n" + "="*80)
print("Generated files:")
print(f"  1. Excel results: {output_excel}")
print(f"  2. Trained models: {models_output_dir}")
print("="*80)
print("\n✅ ANALYSIS COMPLETE!")