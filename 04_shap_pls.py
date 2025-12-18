"""
SHAP Analysis for PLS Model - Simplified Version
Only performs SHAP analysis and visualization, loads pre-trained model from FIVE_MODELS.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Arial']
sns.set_style("whitegrid")

print("="*80)
print("SHAP ANALYSIS FOR PLS MODEL")
print("="*80)

# ============================================================================
# Configuration - Modify paths according to your data location
# ============================================================================
DATA_DIR = "./data"  # Data directory (relative path)
RESULTS_DIR = "./results"  # Output directory for results
MODELS_DIR = "./results/trained_models"  # Trained models directory
SHAP_OUTPUT_DIR = "./results/shap_pls"  # SHAP output directory

# ============================================================================
# Data Loading
# ============================================================================
# Note: Place your data files in the 'data' folder or modify paths accordingly
filepath = f"{DATA_DIR}/selected_features.xlsx"
features_df = pd.read_excel(filepath, sheet_name=0)
target_df = pd.read_excel(filepath, sheet_name=1)

numeric_features = features_df.columns[1:].tolist()
X = features_df[numeric_features].values
y = target_df.iloc[:, 1].values

print(f"\n📊 Dataset: N = {len(X)}, Features = {len(numeric_features)}")

# ============================================================================
# Load Pre-trained PLS Model from FIVE_MODELS.py
# ============================================================================
print("\n⏳ Loading trained PLS model from FIVE_MODELS...")
model_path = f"{MODELS_DIR}/PLS_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"   ✓ Loaded model from: {model_path}")
    print(f"   ✓ Model params: n_components={model.n_components}")
else:
    print(f"   ⚠ Model file not found: {model_path}")
    print(f"   ⚠ Please run FIVE_MODELS.py first to train and save models!")
    raise FileNotFoundError(f"Please run FIVE_MODELS.py first!")

# ============================================================================
# Compute SHAP Values
# ============================================================================
print("\n⏳ Computing SHAP values...")

def pls_predict(X_input):
    return model.predict(X_input).ravel()

background = shap.sample(X, min(100, len(X)), random_state=42)
explainer = shap.KernelExplainer(pls_predict, background)
shap_values = explainer.shap_values(X)

print(f"   ✓ SHAP values computed for {len(X)} samples")

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
# SHAP Visualization
# ============================================================================
def normalize_feature(values):
    vmin, vmax = np.nanpercentile(values, [5, 95])
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    values_clipped = np.clip(values, vmin, vmax)
    return (values_clipped - vmin) / (vmax - vmin)


def plot_shap_ab_combined(shap_vals_array, feat_vals_raw, feat_names_short, model_name, output_dir):
    """Generate SHAP A+B combined plot"""
    mean_abs_shap = np.mean(np.abs(shap_vals_array), axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    
    shap_vals_sorted = shap_vals_array[:, order]
    feat_vals_sorted = feat_vals_raw[:, order]
    feat_names_sorted = [feat_names_short[i] for i in order]
    importance_sorted = mean_abs_shap[order]
    
    feat_vals_norm = np.column_stack([
        normalize_feature(feat_vals_sorted[:, j])
        for j in range(feat_vals_sorted.shape[1])
    ])
    
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={'height_ratios': [1.8, 1], 'hspace': 0.1}
    )
    
    # Panel A: SHAP Scatter Plot
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
            s=50, alpha=0.7, edgecolors='none',
            vmin=0, vmax=1
        )
    
    axA.axhline(0, color='#666666', linestyle='-', linewidth=1, alpha=0.8)
    axA.set_ylabel('SHAP value', fontsize=18, fontweight='bold')
    axA.grid(True, axis='y', alpha=0.3)
    axA.spines['top'].set_visible(False)
    axA.spines['right'].set_visible(False)
    axA.spines['bottom'].set_visible(False)
    axA.set_xticks([])
    
    y_max = max(abs(shap_vals_array.min()), abs(shap_vals_array.max())) * 1.1
    axA.set_ylim(-y_max, y_max)
    
    # Colorbar
    divider_top = 0.95
    cbar_ax = fig.add_axes([0.15, divider_top, 0.7, 0.02])
    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    fig.text(0.5, divider_top + 0.03, 'Feature value',
             ha='center', fontsize=18, fontweight='bold', fontfamily='Arial')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'], fontsize=12, fontweight='bold')
    cbar.ax.tick_params(size=0)
    
    # Panel B: Feature Importance Bar Chart
    bars = axB.bar(
        x_positions, importance_sorted,
        width=0.7, color='teal', alpha=0.8,
        edgecolor='black', linewidth=0.5
    )
    
    axB.set_ylabel('Mean(|SHAP value|)', fontsize=16, fontweight='bold')
    axB.grid(True, axis='y', alpha=0.3)
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)
    
    axB.set_xticks(x_positions)
    axB.set_xticklabels(feat_names_sorted, rotation=45, ha='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    filename = f"{output_dir}/SHAP_{model_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    plt.close()


# ============================================================================
# Generate Plots and Save Results
# ============================================================================
print("\n" + "="*80)
print("GENERATING SHAP VISUALIZATION")
print("="*80)

output_dir = SHAP_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

shap_vals_array = np.array(shap_values)
plot_shap_ab_combined(shap_vals_array, X, feat_names_short, "PLS", output_dir)

# Save to Excel
output_excel = f"{output_dir}/SHAP_Values_PLS.xlsx"
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # SHAP Values
    shap_df = pd.DataFrame(shap_values, columns=feat_names_short)
    shap_df.insert(0, 'Sample', range(1, len(shap_df)+1))
    shap_df.to_excel(writer, sheet_name='PLS_SHAP', index=False)
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': feat_names_short,
        'Original_Name': numeric_features,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    importance_df.to_excel(writer, sheet_name='PLS_Importance', index=False)

print(f"   ✓ Saved: {output_excel}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SHAP ANALYSIS COMPLETE")
print("="*80)

print(f"\n📈 Top 5 Features:")
importance_sorted = sorted(zip(feat_names_short, mean_abs_shap), key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(importance_sorted[:5], 1):
    print(f"   {i}. {feat}: {imp:.4f}")

print(f"\n✓ Output: {output_dir}")
print("="*80)
