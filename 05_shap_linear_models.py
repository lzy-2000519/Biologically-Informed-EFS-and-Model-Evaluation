"""
SHAP Analysis for Three Models: ElasticNet, Bayesian Ridge, and SVR
Based on FIVE_MODELS.py methodology with comprehensive validation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Arial']
sns.set_style("whitegrid")

print("="*80)
print("SHAP ANALYSIS FOR THREE MODELS")
print("="*80)

# ============================================================================
# Configuration - Modify paths according to your data location
# ============================================================================
DATA_DIR = "./data"  # Data directory (relative path)
RESULTS_DIR = "./results"  # Output directory for results
SHAP_OUTPUT_DIR = "./results/shap_linear_models"  # SHAP output directory

# ============================================================================
# Data Loading (Must use the same data as FIVE_MODELS.py!)
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
# Outlier Detection using Cook's Distance (Consistent with FIVE_MODELS)
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
# Use Full Dataset (No train/test split, SHAP analysis on all samples)
# ============================================================================
print(f"\n📊 Using FULL Dataset for SHAP Analysis:")
print(f"   Total samples: {len(X_use)}")

# ============================================================================
# Define Three Model Configurations (Consistent with FIVE_MODELS)
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

print("\n✓ Configured 3 models for SHAP analysis:")
for model_name in models_config.keys():
    print(f"   • {model_name}")

# ============================================================================
# Train Models and Compute SHAP Values
# ============================================================================
shap_results = {}

from sklearn.model_selection import cross_val_score

for model_name, config in models_config.items():
    print("\n" + "="*80)
    print(f"PROCESSING: {model_name} (Full Dataset)")
    print("="*80)
    
    # 1. Hyperparameter Optimization (using CV on full data)
    if config['needs_tuning'] and config['param_grid']:
        print(f"\n⏳ Tuning hyperparameters...")
        inner_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        grid = GridSearchCV(config['model'], config['param_grid'], 
                           cv=inner_cv, scoring='r2', n_jobs=-1)
        grid.fit(X_use, y_use)  # Fit on full data
        model_to_use = grid.best_estimator_
        best_params = grid.best_params_
        print(f"   ✓ Best params: {best_params}")
    else:
        model_to_use = config['model']
        model_to_use.fit(X_use, y_use)  # Fit on full data
        best_params = None
        print(f"   ✓ No tuning needed")
    
    # 2. Evaluate Performance (Full data fit + Cross-validation)
    y_pred_all = model_to_use.predict(X_use)
    full_r2 = r2_score(y_use, y_pred_all)
    full_rmse = np.sqrt(mean_squared_error(y_use, y_pred_all))
    
    # Cross-validation R² (more reliable generalization estimate)
    cv_scores = cross_val_score(model_to_use, X_use, y_use, cv=5, scoring='r2')
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()
    
    print(f"\n📊 Model Performance (Full Dataset: {len(X_use)} samples):")
    print(f"   Full Data R² = {full_r2:.4f}")
    print(f"   Full Data RMSE = {full_rmse:.4f}")
    print(f"   CV R² = {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    
    # 3. Compute SHAP Values (on full data)
    print(f"\n⏳ Computing SHAP values for ALL {len(X_use)} samples...")
    
    if model_name in ['ElasticNet', 'Bayesian Ridge']:
        # Linear models use LinearExplainer (fast)
        explainer = shap.LinearExplainer(model_to_use, X_use, feature_names=numeric_features)
        shap_values = explainer.shap_values(X_use)  # Compute on full data
    else:  # SVR
        # Non-linear models use KernelExplainer
        background = shap.sample(X_use, min(100, len(X_use)), random_state=42)
        explainer = shap.KernelExplainer(model_to_use.predict, background)
        shap_values = explainer.shap_values(X_use)  # Compute on full data
    
    print(f"   ✓ SHAP values computed for {len(X_use)} samples")
    
    # 4. Save Results (full data)
    shap_results[model_name] = {
        'model': model_to_use,
        'shap_values': shap_values,
        'explainer': explainer,
        'X_data': X_use,  # Full data
        'y_data': y_use,
        'y_pred': y_pred_all,
        'full_r2': full_r2,
        'full_rmse': full_rmse,
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'best_params': best_params,
        'n_samples': len(X_use)
    }

# ============================================================================
# Feature Name Abbreviation Mapping
# ============================================================================
feature_name_mapping = {
    'D-Phenylalanine': 'D-Phe',
    'D-Ornithine': 'D-Orn',
    'Hippuric acid': 'HA',
    'L-Alanine': 'L-Ala',
    'Ethanolamine': 'EA',
    'D-Glutamine': 'D-Gln',
    'Kynurenic acid': 'KYNA',
    'N-Acetylputrescine': 'NAP',
    'N6,N6,N6-Trimethyl-L-lysine': 'TML',
    'Serotonin': '5-HT'
}

feat_names_short = [feature_name_mapping.get(f, f) for f in numeric_features]

# ============================================================================
# SHAP Visualization Functions (A+B Combined Plot Style)
# ============================================================================
def normalize_feature(values):
    """Normalize feature values (using 5%-95% percentiles to avoid extremes)"""
    vmin, vmax = np.nanpercentile(values, [5, 95])
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    values_clipped = np.clip(values, vmin, vmax)
    return (values_clipped - vmin) / (vmax - vmin)


def plot_shap_ab_combined(shap_vals_array, feat_vals_raw, feat_names_short, model_name, output_dir):
    """
    Generate SHAP A+B Combined Plot
    - Panel A: SHAP scatter plot (top, shows SHAP value distribution, color indicates feature value)
    - Panel B: Feature importance bar chart (bottom)
    """
    # Calculate feature importance and sort
    mean_abs_shap = np.mean(np.abs(shap_vals_array), axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    
    # Sort by importance
    shap_vals_sorted = shap_vals_array[:, order]
    feat_vals_sorted = feat_vals_raw[:, order]
    feat_names_sorted = [feat_names_short[i] for i in order]
    importance_sorted = mean_abs_shap[order]
    
    # Normalize feature values
    feat_vals_norm = np.column_stack([
        normalize_feature(feat_vals_sorted[:, j])
        for j in range(feat_vals_sorted.shape[1])
    ])
    
    # Create subplots
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={'height_ratios': [1.8, 1], 'hspace': 0.1}
    )
    
    # === Panel A: SHAP Summary Plot ===
    n_features = len(feat_names_sorted)
    x_positions = np.arange(n_features)
    np.random.seed(42)
    
    for j, x_pos in enumerate(x_positions):
        shap_values_feat = shap_vals_sorted[:, j]
        feature_values_norm = feat_vals_norm[:, j]
        jitter = np.random.normal(0, 0.06, size=len(shap_values_feat))
        x_jittered = x_pos + jitter
        
        scatter = axA.scatter(
            x_jittered, shap_values_feat,
            c=feature_values_norm,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='none',
            vmin=0, vmax=1
        )
    
    axA.axhline(0, color='#666666', linestyle='-', linewidth=1, alpha=0.8)
    axA.set_ylabel('SHAP value', fontsize=18, fontweight='bold')
    axA.grid(True, axis='y', alpha=0.3)
    axA.spines['top'].set_visible(False)
    axA.spines['right'].set_visible(False)
    axA.spines['bottom'].set_visible(False)
    axA.set_xticks([])
    
    # Y-axis range (dynamic adjustment)
    y_max = max(abs(shap_vals_array.min()), abs(shap_vals_array.max())) * 1.1
    axA.set_ylim(-y_max, y_max)
    
    # Add colorbar at top
    divider_top = 0.95
    cbar_ax = fig.add_axes([0.15, divider_top, 0.7, 0.02])
    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    fig.text(0.5, divider_top + 0.03, 'Feature value',
             ha='center', fontsize=18, fontweight='bold', fontfamily='Arial')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'], fontsize=12, fontweight='bold')
    cbar.ax.tick_params(size=0)
    
    # === Panel B: Feature Importance ===
    bars = axB.bar(
        x_positions, importance_sorted,
        width=0.7,
        color='teal',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    axB.set_ylabel('Mean(|SHAP value|)', fontsize=16, fontweight='bold')
    axB.grid(True, axis='y', alpha=0.3)
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)
    axB.spines['bottom'].set_visible(True)
    axB.spines['left'].set_visible(True)
    
    for spine_name, spine in axB.spines.items():
        if spine_name not in ['top', 'right']:
            spine.set_linewidth(1.5)
            spine.set_color('black')
    
    axB.set_xticks(x_positions)
    axB.set_xticklabels(feat_names_sorted, rotation=45, ha='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    filename = f"{output_dir}/SHAP_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()


# ============================================================================
# Generate SHAP Visualizations
# ============================================================================
print("\n" + "="*80)
print("GENERATING SHAP VISUALIZATIONS")
print("="*80)

output_dir = SHAP_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

for model_name, result in shap_results.items():
    print(f"\n📊 Generating SHAP plot for {model_name} (Full Dataset: {result['n_samples']} samples)...")
    
    shap_vals = result['shap_values']
    X_data_vals = result['X_data']  # Use full data
    shap_vals_array = np.array(shap_vals)
    
    # Generate SHAP A+B combined plot
    plot_shap_ab_combined(
        shap_vals_array, X_data_vals, feat_names_short, model_name, output_dir
    )

# ============================================================================
# Save SHAP Values to Excel
# ============================================================================
print("\n" + "="*80)
print("SAVING SHAP VALUES TO EXCEL")
print("="*80)

output_excel = f"{output_dir}/SHAP_Values_Three_Models.xlsx"

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    for model_name, result in shap_results.items():
        # SHAP Values (all samples)
        shap_df = pd.DataFrame(
            result['shap_values'],
            columns=feat_names_short
        )
        shap_df.insert(0, 'Sample', range(1, len(shap_df)+1))
        shap_df.to_excel(writer, sheet_name=f'{model_name}_SHAP', index=False)
        
        # Feature Importance
        mean_abs_shap = np.mean(np.abs(result['shap_values']), axis=0)
        importance_df = pd.DataFrame({
            'Feature': feat_names_short,
            'Original_Name': numeric_features,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        importance_df.to_excel(writer, sheet_name=f'{model_name}_Importance', index=False)
    
    # Model Performance Summary
    perf_data = []
    for model_name, result in shap_results.items():
        perf_data.append({
            'Model': model_name,
            'N_Samples': result['n_samples'],
            'Full_Data_R2': result['full_r2'],
            'Full_Data_RMSE': result['full_rmse'],
            'CV_R2_Mean': result['cv_r2_mean'],
            'CV_R2_Std': result['cv_r2_std'],
            'Best_Params': str(result['best_params']) if result['best_params'] else 'None'
        })
    perf_df = pd.DataFrame(perf_data)
    perf_df.to_excel(writer, sheet_name='Model_Performance', index=False)

print(f"✓ SHAP values saved: {output_excel}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("SHAP ANALYSIS COMPLETE (Full Dataset)")
print("="*80)

print(f"\n📊 Summary (Full Dataset: {len(X_use)} samples):")
for model_name, result in shap_results.items():
    print(f"\n{model_name}:")
    print(f"   Full Data R² = {result['full_r2']:.4f}")
    print(f"   Full Data RMSE = {result['full_rmse']:.4f}")
    print(f"   CV R² = {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
    if result['best_params']:
        print(f"   Best params: {result['best_params']}")

print(f"\n✓ All visualizations saved to: {output_dir}")
print(f"✓ SHAP values saved to: {output_excel}")

print("\n" + "="*80)
print("FILES GENERATED:")
print("="*80)
print(f"  1. SHAP_ElasticNet.png")
print(f"  2. SHAP_Bayesian_Ridge.png")
print(f"  3. SHAP_SVR.png")
print(f"  4. SHAP_Values_Three_Models.xlsx")
print(f"\nTotal: 3 PNG files (A+B combined) + 1 Excel file")
print("="*80)
