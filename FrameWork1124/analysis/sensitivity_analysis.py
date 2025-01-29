import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def is_pareto_efficient(costs, latencies):
    """
    Encontra os pontos não dominados (fronteira de Pareto)
    Um ponto é não dominado se não existe outro ponto que seja melhor em ambas as dimensões
    """
    points = np.column_stack((costs, latencies))
    is_efficient = np.ones(points.shape[0], dtype=bool)
    
    for i, point in enumerate(points):
        if is_efficient[i]:
            # Um ponto domina outro se for estritamente menor em pelo menos uma dimensão
            # e menor ou igual em todas as outras
            dominates = (points < point).any(axis=1) & (points <= point).all(axis=1)
            # Remove o próprio ponto da comparação
            dominates[i] = False
            # Marca como falso os pontos que são dominados
            is_efficient[i] = not dominates.any()
    
    return is_efficient

def filter_close_points(costs, latencies, threshold=0.01):
    """
    Remove pontos que estão muito próximos uns dos outros.
    threshold: valor relativo (%) da distância máxima considerada como "próxima"
    """
    points = np.column_stack((costs, latencies))
    
    # Normalizar os pontos para que as distâncias sejam comparáveis
    costs_range = np.ptp(costs)  # max - min
    latencies_range = np.ptp(latencies)  # max - min
    normalized_points = np.column_stack((
        (costs - np.min(costs)) / costs_range,
        (latencies - np.min(latencies)) / latencies_range
    ))
    
    # Inicializar máscara de pontos a manter
    keep_mask = np.ones(len(points), dtype=bool)
    
    for i in range(len(normalized_points)):
        if keep_mask[i]:
            # Calcular distâncias euclidianas para todos os outros pontos
            distances = np.sqrt(np.sum((normalized_points - normalized_points[i])**2, axis=1))
            
            # Marcar pontos próximos para remoção (exceto o ponto atual)
            close_points = distances < threshold
            close_points[i] = False  # Manter o ponto atual
            
            # Atualizar máscara
            keep_mask = keep_mask & ~close_points
    
    return keep_mask

def analyze_sensitivity_results(sensitivity_dir, pop_name, window_size, step_size, output_dir='analysis_results/sensitivity'):
    """
    Analyze results from sensitivity analysis
    """
    # Criar diretório base e subdiretório específico para a população
    base_output_dir = os.path.join(output_dir, f'{pop_name}/{window_size}x{step_size}')
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Read combined results
    df = pd.read_csv(os.path.join(sensitivity_dir, f'{pop_name}_{window_size}x{step_size}_all_weights_results.csv'))
    
    # Sort by cost_weight for proper plotting
    df = df.sort_values('cost_weight')
    
    # Create plots for each metric
    metrics = {
        'total_model_cost': 'Total Cost',
        'total_model_latency': 'Total Latency (s)',
        'accuracy': 'Accuracy',
        'f1_score': 'F1 Score'
    }
    
    # Extrair nomes únicos dos algoritmos dos dados
    algorithms = sorted(list(set([model.split('_')[0] for model in df['model'].unique()])))
    
    # Lista expandida de marcadores disponíveis
    available_markers = [
        'o', '*', 'D', 's', '^', 'v', 'P', 'X', '<', '>', 'p', 'h', 'H', 
        '8', '+', 'x', 'd', '|', '_', '1', '2', '3', '4'
    ]
    
    # Verificar se temos marcadores suficientes
    if len(algorithms) > len(available_markers):
        raise ValueError(f"Número de algoritmos ({len(algorithms)}) excede o número de marcadores disponíveis ({len(available_markers)})")
    
    # Criar dicionários dinâmicos para marcadores e cores
    markers = {
        algo: marker for algo, marker in zip(algorithms, available_markers)
    }
    
    # Criar paleta de cores usando colormap
    colors = {
        algo: color for algo, color in zip(
            algorithms,
            plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
        )
    }
    
    # Print para debug
    print("Algoritmos encontrados:", algorithms)
    print("Marcadores atribuídos:", markers)
    
    # Plot trends
    for metric, metric_label in metrics.items():
        plt.figure(figsize=(12, 6))
        for model in df['model'].unique():
            model_base = model.split('_')[0]  # Pegar nome base do modelo
            model_data = df[df['model'] == model]
            plt.plot(model_data['cost_weight'].to_numpy(), 
                    model_data[metric].to_numpy(),
                    marker=markers[model_base],
                    color=colors[model_base],
                    label=model,
                    markersize=8,
                    alpha=0.7)
        
        plt.xlabel('Cost Weight', fontsize=12)
        plt.ylabel(metric_label, fontsize=12)
        plt.title(f'{metric_label} vs Cost Weight', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  borderaxespad=0.,
                  fontsize=10,
                  frameon=True)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_output_dir, f'{metric}_trend.png'), 
                    dpi=300, 
                    bbox_inches='tight')
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
    # Create like this: Pop1/4x4/relative_changes_heatmap.png
    plt.savefig(os.path.join(base_output_dir, 'relative_changes_heatmap.png'))
    plt.close()
    
    # Generate summary statistics
    summary = df.groupby(['model', 'cost_weight']).agg({
        metric: ['mean', 'std'] for metric in metrics
    }).round(4)
    
    summary.to_csv(os.path.join(base_output_dir, 'sensitivity_summary.csv'))
    
    # Criar diretórios específicos para cada modelo
    for model in df['model'].unique():
        model_dir = os.path.join(base_output_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        
        # Dados específicos do modelo
        model_data = df[df['model'] == model]
        
        # Criar scatter plot de trade-off custo x latência para cada modelo
        plt.figure(figsize=(12, 8))
        
        weights = model_data['cost_weight'].unique()
        colors_weight = plt.cm.viridis(np.linspace(0, 1, len(weights)))
        
        for weight, color in zip(weights, colors_weight):
            weight_data = model_data[model_data['cost_weight'] == weight]
            plt.scatter(weight_data['total_model_cost'].to_numpy(),
                       weight_data['total_model_latency'].to_numpy(),
                       label=f'w={weight}',
                       color=color,
                       marker=markers[model.split('_')[0]],  # Usar marcador específico
                       alpha=0.7,
                       s=100)
            
            # Anotações mais claras
            for x, y in zip(weight_data['total_model_cost'],
                          weight_data['total_model_latency']):
                plt.annotate(f'w={weight}',
                           (x, y),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(facecolor='white', 
                                   edgecolor='none', 
                                   alpha=0.7))
        
        plt.xlabel('Custo Total')
        plt.ylabel('Latência Total (s)')
        plt.title(f'Trade-off: Custo vs Latência - Modelo {model}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'cost_latency_tradeoff.png'))
        plt.close()
    
    # Criar plot da fronteira de Pareto global
    plt.figure(figsize=(12, 8))
    
    # Coletar todos os pontos de todos os modelos
    all_costs = df['total_model_cost'].to_numpy()
    all_latencies = df['total_model_latency'].to_numpy()
    
    # Encontrar pontos não dominados considerando todos os pontos
    efficient_mask = is_pareto_efficient(all_costs, all_latencies)
    
    # Criar DataFrame apenas com pontos não dominados
    pareto_df = df[efficient_mask].copy()
    
    # Após encontrar os pontos de Pareto, filtrar pontos próximos
    pareto_mask = is_pareto_efficient(
        df['total_model_cost'].to_numpy(),
        df['total_model_latency'].to_numpy()
    )
    
    # Aplicar filtro de proximidade apenas nos pontos de Pareto
    pareto_points = df[pareto_mask]
    keep_mask = filter_close_points(
        pareto_points['total_model_cost'].to_numpy(),
        pareto_points['total_model_latency'].to_numpy(),
        threshold=0.05  # ajuste este valor conforme necessário (5% da distância máxima)
    )
    
    # Criar DataFrame final apenas com pontos filtrados
    pareto_df = pareto_points[keep_mask].copy()
    
    # Extrair o nome do pop do modelo (assumindo formato "Pop1_1_1", "Pop2_4_4", etc)
    pareto_df['pop'] = pareto_df['model'].apply(lambda x: x.split('_')[0])
    
    # Reordenar as colunas para melhor visualização
    cols = ['pop', 'model', 'cost_weight', 'total_model_cost', 'total_model_latency', 'accuracy', 'f1_score']
    pareto_df = pareto_df[cols]
    
    # Plotar pontos não dominados, coloridos por modelo
    models = pareto_df['model'].unique()
    model_colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for model, color in zip(models, model_colors):
        model_data = pareto_df[pareto_df['model'] == model]
        plt.scatter(model_data['total_model_cost'],
                   model_data['total_model_latency'],
                   label=model,
                   color=color,
                   marker=markers[model.split('_')[0]],
                   alpha=0.7,
                   s=100)
        
        # Melhorar anotações
        for _, row in model_data.iterrows():
            plt.annotate(f'w={row["cost_weight"]}',
                        (row['total_model_cost'], 
                         row['total_model_latency']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(facecolor='white', 
                                edgecolor='none', 
                                alpha=0.7))
    
    plt.xlabel('Custo Total')
    plt.ylabel('Latência Total (s)')
    plt.title('Fronteira de Pareto Global: Custo vs Latência')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, 'pareto_frontier_global.png'))
    plt.close()
    
    # Salvar dados dos pontos não dominados
    pareto_df.to_csv(os.path.join(base_output_dir, 'pareto_frontier_points.csv'), index=False)
    
    return summary

if __name__ == "__main__":
    sensitivity_dir = "../results/sensitivity_analysis"
    pop_name = "Pop2"
    window_size = 4
    step_size = 4   
    summary = analyze_sensitivity_results(sensitivity_dir, pop_name, window_size, step_size)
    print("Analysis complete. Check the analysis_results/sensitivity directory for outputs.") 