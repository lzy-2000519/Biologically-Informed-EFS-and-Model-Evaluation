import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

plt.rcParams['font.family'] = ['arial']

# Configuration
DATA_DIR = "./data"
VIZ_DIR = "./results/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# Load data and set first column as index (feature names)
file_path = f"{DATA_DIR}/Lasso_feature_frequency.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df.set_index(df.columns[0], inplace=True)  # Set first column as feature names

# Get selection frequency (column 2) and coefficient (column 3)
frequency_col = df.columns[0]  # Column 2: Feature selection frequency
importance_col = df.columns[1]  # Column 3: Coefficient

# Filter out rows with frequency = 0
df_filtered = df[df[frequency_col] != 0].copy()

# Sort by selection frequency in descending order
df_sorted = df_filtered.sort_values(by=frequency_col, ascending=False)


def plot_feature_frequency(importances, frequencies, model_name, top_n=20, save_path=None,
                           bar_color=(70, 130, 180)):
    """
    Plot horizontal bar chart:
    - Bar length: Feature selection frequency
    - Bar color: Uniform color via RGB tuple (R, G, B), range 0-255

    Parameters:
    - importances: Sorted DataFrame
    - frequencies: Frequency data
    - model_name: Model name
    - top_n: Number of top features to display (default 20)
    - save_path: Save path (if None, only display)
    - bar_color: RGB color tuple, default (70, 130, 180) steel blue
                 Common colors: (70, 130, 180) steel blue, (255, 99, 71) tomato red,
                         (34, 139, 34) forest green, (255, 165, 0) orange
    """
    plt.figure(figsize=(6, 8))

    # Select top N features
    sorted_importances = importances.head(top_n)
    sorted_frequencies = frequencies.head(top_n)

    # Convert RGB values to 0-1 range (required by matplotlib)
    normalized_color = tuple(c / 255.0 for c in bar_color)

    # Plot each bar
    for i, feat in enumerate(sorted_importances.index):
        freq = sorted_frequencies[feat]  # Get selection frequency
        plt.barh(i, freq, color=normalized_color)

    # Use index (feature names) as y-axis labels
    plt.yticks(range(len(sorted_importances)), sorted_importances.index, fontsize=12, weight='bold')
    plt.xlabel('Selection Frequency', fontsize=12, weight='bold')
    plt.title(f'Feature Frequency - {model_name}', fontsize=15, weight='bold')
    plt.ylabel('Features', fontsize=12, weight='bold')  # Add Y-axis label with font size 12
    plt.xticks(fontsize=12)
    plt.ylim(-0.5, len(sorted_importances) - 0.5)
    # Invert y-axis so most important features appear at top
    plt.gca().invert_yaxis()


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


# Call plotting function
# bar_color parameter uses RGB tuple, range 0-255
plot_feature_frequency(df_sorted, df_sorted[frequency_col], "LASSO",
                       save_path=f"{VIZ_DIR}/lasso_Feature_Frequency.png",
                       bar_color=(70, 130, 180))