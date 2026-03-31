import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_aggregated_results(results_list, save_dir):
    """Takes the flat list of dicts from the Automaton, averages over seeds, and plots."""
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results_list)
    
    # We want a separate plot for each Dataset
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        df_sub = df[df['dataset'] == dataset]
        
        # Plot 1: Accuracy Comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_sub, x='model', y='accuracy', hue='precision', capsize=.1, errorbar='sd')
        plt.title(f'Accuracy by Precision on {dataset.upper()} (Averaged over seeds)')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.legend(title='Precision')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset}_accuracy_comparison.png"), dpi=300)
        plt.close()

        # Plot 2: Latency Comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_sub, x='model', y='latency_ms', hue='precision', capsize=.1, errorbar='sd')
        plt.title(f'Inference Latency on {dataset.upper()} (Averaged over seeds)')
        plt.ylabel('Latency (ms/sample)')
        plt.legend(title='Precision')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset}_latency_comparison.png"), dpi=300)
        plt.close()
        
        # Plot 3: ECE (Calibration) Comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_sub, x='model', y='ece', hue='precision', capsize=.1, errorbar='sd')
        plt.title(f'Expected Calibration Error on {dataset.upper()} (Lower is Better)')
        plt.ylabel('ECE')
        plt.legend(title='Precision')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset}_ece_comparison.png"), dpi=300)
        plt.close()
        
    print(f"Aggregated plots saved successfully to the '{save_dir}' folder.")