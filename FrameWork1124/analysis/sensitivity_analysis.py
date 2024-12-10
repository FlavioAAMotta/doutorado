import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_sensitivity_results(sensitivity_dir, output_dir='analysis_results/sensitivity'):
    """
    Analyze results from sensitivity analysis
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read combined results
    df = pd.read_csv(os.path.join(sensitivity_dir, 'all_weights_results.csv'))
    
    # Sort by cost_weight for proper plotting
    df = df.sort_values('cost_weight')
    
    # Create plots for each metric
    metrics = {
        'total_model_cost': 'Total Cost',
        'total_model_latency': 'Total Latency (s)',
        'accuracy': 'Accuracy',
        'f1_score': 'F1 Score'
    }
    
    # Plot trends
    for metric, metric_label in metrics.items():
        plt.figure(figsize=(12, 6))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            plt.plot(model_data['cost_weight'].to_numpy(), 
                    model_data[metric].to_numpy(),
                    marker='o', 
                    label=model)
        
        plt.xlabel('Cost Weight')
        plt.ylabel(metric_label)
        plt.title(f'{metric_label} vs Cost Weight')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_trend.png'))
        plt.close()
    
    # Create heatmap of relative changes
    plt.figure(figsize=(15, 10))
    for idx, metric in enumerate(metrics.keys(), 1):
        plt.subplot(2, 2, idx)
        pivot_data = df.pivot(index='model', 
                            columns='cost_weight', 
                            values=metric)
        # Normalize relative to weight=50
        baseline = pivot_data[50]
        relative_change = 100 * (pivot_data.subtract(baseline, axis=0)
                                .divide(baseline, axis=0))
        
        sns.heatmap(relative_change, 
                   cmap='RdYlBu_r',
                   center=0,
                   annot=True,
                   fmt='.1f')
        plt.title(f'Relative Change in {metrics[metric]} (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relative_changes_heatmap.png'))
    plt.close()
    
    # Generate summary statistics
    summary = df.groupby(['model', 'cost_weight']).agg({
        metric: ['mean', 'std'] for metric in metrics
    }).round(4)
    
    summary.to_csv(os.path.join(output_dir, 'sensitivity_summary.csv'))
    
    # Criar diretórios específicos para cada modelo
    for model in df['model'].unique():
        model_dir = os.path.join(output_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        
        # Dados específicos do modelo
        model_data = df[df['model'] == model]
        
        # Criar scatter plot de trade-off custo x latência para cada modelo
        plt.figure(figsize=(12, 8))
        
        # Criar um mapa de cores baseado nos pesos
        weights = model_data['cost_weight'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
        
        for weight, color in zip(weights, colors):
            weight_data = model_data[model_data['cost_weight'] == weight]
            scatter = plt.scatter(weight_data['total_model_cost'].to_numpy(),
                                weight_data['total_model_latency'].to_numpy(),
                                label=f'w={weight}',
                                color=color,
                                alpha=0.7,
                                s=100)
            
            # Adicionar anotações para cada ponto
            for x, y in zip(weight_data['total_model_cost'],
                          weight_data['total_model_latency']):
                plt.annotate(f'w={weight}',
                            (x, y),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8)
        
        plt.xlabel('Custo Total')
        plt.ylabel('Latência Total (s)')
        plt.title(f'Trade-off: Custo vs Latência - Modelo {model}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'cost_latency_tradeoff.png'))
        plt.close()
    
    return summary

if __name__ == "__main__":
    sensitivity_dir = "results/sensitivity_analysis"
    summary = analyze_sensitivity_results(sensitivity_dir)
    print("Analysis complete. Check the analysis_results/sensitivity directory for outputs.") 