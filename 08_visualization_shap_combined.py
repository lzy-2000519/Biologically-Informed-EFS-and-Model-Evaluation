import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import shap
import warnings

warnings.filterwarnings('ignore')

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.monospace'] = ['Arial']
mpl.rcParams['font.family'] = 'Arial'

print("Generating optimized SHAP A+B combined plot...")

# Configuration
DATA_DIR = "./data"
VIZ_DIR = "./results/visualizations"
import os
os.makedirs(VIZ_DIR, exist_ok=True)

# Data Preparation
filepath = f"{DATA_DIR}/selected_features.xlsx"
features_df = pd.read_excel(filepath, sheet_name=0)
target_df = pd.read_excel(filepath, sheet_name=1)

# Auto-detect numeric feature columns (skip sample ID column)
numeric_features = features_df.columns[1:].tolist()

X = features_df[numeric_features]
y = target_df.iloc[:, 1]  # Read column 2 as target variable

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, shuffle=True
)

# Quick Hyperparameter Optimization
alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
l1_ratio_values = [0.1, 0.3, 0.5, 0.7, 0.9]

inner_cv = KFold(n_splits=5, shuffle=True, random_state=123)
best_score = -np.inf
best_params = {}

for alpha in alpha_values:
    for l1_ratio in l1_ratio_values:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                      max_iter=2000, random_state=42))
        ])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=inner_cv, scoring='r2')
        mean_score = cv_scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}

print(f"Best params: alpha = {best_params['alpha']}, l1_ratio = {best_params['l1_ratio']}")

# Build and Train Best Model
best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNet(
        alpha=best_params['alpha'],
        l1_ratio=best_params['l1_ratio'],
        max_iter=2000,
        random_state=42
    ))
])

best_pipeline.fit(X_train, y_train)

# Get Model and Data
elasticnet_model = best_pipeline.named_steps['elasticnet']
X_train_scaled = best_pipeline.named_steps['scaler'].transform(X_train)
X_test_scaled = best_pipeline.named_steps['scaler'].transform(X_test)

# Compute SHAP Values
explainer = shap.LinearExplainer(elasticnet_model, X_train_scaled, feature_names=numeric_features)
shap_values_test = explainer.shap_values(X_test_scaled)

# Feature Name Abbreviation Mapping
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

print("Generating optimized SHAP A+B combined plot...")

# ========== SHAP A+B Combined Plot (Optimized Version) ==========
# Data Preparation
shap_vals = np.array(shap_values_test)  # (n_samples, n_features)
feat_vals_raw = X_test[numeric_features].values  # Raw feature values for color mapping
feat_names = [feature_name_mapping.get(f, f) for f in numeric_features]

# Calculate feature importance and sort (high to low, important on left)
mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
order = np.argsort(mean_abs_shap)[::-1]  # Descending order, high→low importance

# Sort all data by importance
shap_vals_sorted = shap_vals[:, order]
feat_vals_sorted = feat_vals_raw[:, order]
feat_names_sorted = [feat_names[i] for i in order]
importance_sorted = mean_abs_shap[order]


# Feature value normalization (using percentile range to avoid extreme values)
def normalize_feature(values):
    # Use 5% and 95% percentiles instead of min/max to avoid extreme outliers
    vmin, vmax = np.nanpercentile(values, [5, 95])
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    # Clip values to percentile range
    values_clipped = np.clip(values, vmin, vmax)
    return (values_clipped - vmin) / (vmax - vmin)


feat_vals_norm = np.column_stack([
    normalize_feature(feat_vals_sorted[:, j])
    for j in range(feat_vals_sorted.shape[1])
])

# Set plot parameters
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.figsize': (12, 10)
})

# Create subplots
fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(12, 10),
    gridspec_kw={'height_ratios': [1.8, 1], 'hspace': 0.1}
)

# === Panel A: SHAP Summary Plot (Scatter) ===
n_features = len(feat_names_sorted)
x_positions = np.arange(n_features)

# Set random seed for consistent jitter
np.random.seed(42)

# Plot scatter points for each feature
for j, x_pos in enumerate(x_positions):
    shap_values_feat = shap_vals_sorted[:, j]
    feature_values_norm = feat_vals_norm[:, j]

    # Add horizontal jitter to avoid overlap
    jitter = np.random.normal(0, 0.06, size=len(shap_values_feat))
    x_jittered = x_pos + jitter

    # Plot scatter points, color mapped to normalized feature values
    scatter = axA.scatter(
        x_jittered, shap_values_feat,
        c=feature_values_norm,
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolors='none',
        vmin=0, vmax=1
    )

# Style Panel A
axA.axhline(0, color='#666666', linestyle='-', linewidth=1, alpha=0.8)
axA.set_ylabel('SHAP value', fontsize=18,fontweight='bold')
axA.grid(True, axis='y', alpha=0.3)
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)
axA.spines['bottom'].set_visible(False)
axA.set_xticks([])  # Hide x-axis ticks

# Set Y-axis range to -0.3 to 0.3
axA.set_ylim(-0.3, 0.3)

# Add Panel A label
# axA.text(-0.08, 1.02, 'A', transform=axA.transAxes,
#          fontsize=16, fontweight='bold', va='bottom')

# Add top colorbar
divider_top = 0.95
cbar_ax = fig.add_axes([0.15, divider_top, 0.7, 0.02])
cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')

# 用fig.text在colorbar上面添加标签
fig.text(0.5, divider_top + 0.03, 'Feature value',
         ha='center', fontsize=18, fontweight='bold', fontfamily='Arial')

cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Low', 'High'], fontsize=12, fontweight='bold')
cbar.ax.tick_params(size=0)

# === Panel B: Feature Importance (Bar Chart) ===
bars = axB.bar(
    x_positions, importance_sorted,
    width=0.7,
    color='teal',
    alpha=0.8,
    edgecolor='black',
    linewidth=0.5
)

# Style Panel B
axB.set_ylabel('Mean(|SHAP value|)', fontsize=16,fontweight='bold')
axB.grid(True, axis='y', alpha=0.3)

# Set Panel B borders (hide top and right)
axB.spines['top'].set_visible(False)  # Hide top border
axB.spines['right'].set_visible(False)  # Hide right border
axB.spines['bottom'].set_visible(True)
axB.spines['left'].set_visible(True)

# Set border line width and color
for spine_name, spine in axB.spines.items():
    if spine_name not in ['top', 'right']:  # Skip top and right borders
        spine.set_linewidth(1.5)
        spine.set_color('black')

# Set x-axis labels
axB.set_xticks(x_positions)
axB.set_xticklabels(feat_names_sorted, rotation=45, ha='center', fontsize=20, fontweight='bold')

# Add Panel B label
# axB.text(-0.08, 1.02, 'B', transform=axB.transAxes,
#          fontsize=16, fontweight='bold', va='bottom')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Leave space for top colorbar

# Save figure
output_path = f"{VIZ_DIR}/shap_ab_optimized.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print("Optimized SHAP A+B combined plot generated!")
print(f"Saved as: {output_path}")

# Print feature importance ranking
print(f"\nFeature Importance Ranking (high to low):")
print(f"{'Rank':<4} {'Abbrev':<12} {'Original Name':<25} {'Importance':<10}")
print("-" * 55)

for i, (short_name, original_name, importance) in enumerate(
        zip(feat_names_sorted, [numeric_features[j] for j in order], importance_sorted), 1
):
    print(f"{i:2d}.  {short_name:<12} {original_name:<25} {importance:8.4f}")

print(f"\nData Info:")
print(f"  Samples: {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})")
print(f"  Features: {len(numeric_features)}")
print(f"  SHAP value range: [{shap_vals.min():.3f}, {shap_vals.max():.3f}]")

print(f"\nPlot Features:")
print(f"  ✓ Panel A: Scatter plot showing SHAP values, color indicates feature value")
print(f"  ✓ Panel B: Bar chart showing feature importance")
print(f"  ✓ Features sorted by importance (left to right)")
print(f"  ✓ Jitter added to avoid point overlap")
print(f"  ✓ Viridis colormap: purple (low) → yellow (high)")