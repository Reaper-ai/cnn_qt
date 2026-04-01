import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# GLOBAL THEME SETTINGS
# ==========================================
def apply_light_theme():
    """Applies your crisp, clean light theme to all plots."""
    plt.style.use('default') 
    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.size": 11,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white"
    })
    
    # Return the exact color palette requested
    return {
        "FP32": "#2196F3",      # Blue
        "FP16": "#FF9800",      # Orange
        "INT8_PTQ": "#E91E63",  # Pink
        "INT8_QAT": "#4CAF50"   # Green
    }

# ==========================================
# SEPARATE PLOTS (ACCURACY, LATENCY, ECE)
# ==========================================
def plot_aggregated_results(results_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results_list)
    
    # Apply theme and get our custom colors
    custom_palette = apply_light_theme()
    
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        df_sub = df[df['dataset'] == dataset]
        
        # ---------------------------------------------------------
        # Plot 1: Accuracy Comparison (DYNAMIC SCALE)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=df_sub, x='model', y='accuracy', hue='precision', 
            palette=custom_palette, edgecolor="white", linewidth=1.2, 
            capsize=.1, err_kws={'color': '#333333'}
        )
        
        plt.title(f'Accuracy by Precision on {dataset.upper()}', fontsize=14, fontweight="bold", pad=15)
        plt.ylabel('Accuracy (%)', fontweight="bold")
        plt.xlabel('Model Architecture', fontweight="bold")
        
        # Dynamic y-limits to zoom in on the differences
        min_acc = df_sub['accuracy'].min()
        max_acc = df_sub['accuracy'].max()
        padding = max(1.0, (max_acc - min_acc) * 0.5)
        plt.ylim(max(0, min_acc - padding), min(100, max_acc + padding)) 
        
        plt.legend(title='Precision', bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset}_accuracy_comparison.png"), bbox_inches="tight")
        plt.close()

        # ---------------------------------------------------------
        # Plot 2: Latency Comparison
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=df_sub, x='model', y='latency_ms', hue='precision', 
            palette=custom_palette, edgecolor="white", linewidth=1.2, 
            capsize=.1, err_kws={'color': '#333333'}
        )
        
        plt.title(f'Inference Latency on {dataset.upper()}', fontsize=14, fontweight="bold", pad=15)
        plt.ylabel('Latency (ms/sample)', fontweight="bold")
        plt.xlabel('Model Architecture', fontweight="bold")
        
        # Dynamic y-limits
        min_lat = df_sub['latency_ms'].min()
        max_lat = df_sub['latency_ms'].max()
        padding_lat = max(0.1, (max_lat - min_lat) * 0.2)
        plt.ylim(max(0, min_lat - padding_lat), max_lat + padding_lat)
        
        plt.legend(title='Precision', bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset}_latency_comparison.png"), bbox_inches="tight")
        plt.close()
        
        # ---------------------------------------------------------
        # Plot 3: ECE (Calibration) Comparison
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=df_sub, x='model', y='ece', hue='precision', 
            palette=custom_palette, edgecolor="white", linewidth=1.2, 
            capsize=.1, err_kws={'color': '#333333'}
        )
        
        plt.title(f'Expected Calibration Error on {dataset.upper()}', fontsize=14, fontweight="bold", pad=15)
        plt.ylabel('ECE (Lower is Better)', fontweight="bold")
        plt.xlabel('Model Architecture', fontweight="bold")
        
        # Dynamic y-limits
        min_ece = df_sub['ece'].min()
        max_ece = df_sub['ece'].max()
        padding_ece = max(0.005, (max_ece - min_ece) * 0.2)
        plt.ylim(max(0, min_ece - padding_ece), max_ece + padding_ece)
        
        plt.legend(title='Precision', bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset}_ece_comparison.png"), bbox_inches="tight")
        plt.close()

# ==========================================
# EXECUTION LOGIC
# ==========================================
def main():
    metrics_file = "metrics.json"
    save_dir = "fig"
    
    if not os.path.exists(metrics_file):
        print(f"Error: Could not find '{metrics_file}'. Make sure you are in the project root.")
        return

    print(f"Loading data from {metrics_file}...")
    with open(metrics_file, "r") as f:
        results_data = json.load(f)

    print("Generating separated plots with the custom light theme...")
    plot_aggregated_results(results_data, save_dir)
    print(f"Success! Check the '{save_dir}' folder for the separate images.")

if __name__ == "__main__":
    main()