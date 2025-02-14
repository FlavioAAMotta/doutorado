import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_pareto_frontier(pop_name):
    # Configurar o estilo do seaborn
    sns.set_style("whitegrid")
    
    # Definir o caminho do arquivo
    csv_path = f"analysis_results/sensitivity/{pop_name}/4x4/pareto_frontier_points.csv"
    
    # Ler o arquivo CSV
    df = pd.read_csv(csv_path)
    
    # Criar o plot com estilo melhorado
    plt.figure(figsize=(10, 8))
    
    # Definir marcadores e cores para cada modelo
    markers = {
        'AHL': 'o', 'ONL': '*', 'SVML': 'D', 'SV-Grid': 's', 
        'SV': '^', 'SVMR': 'v'
    }
    
    colors = {
        'AHL': '#1f77b4', 'ONL': '#ff7f0e', 'SVML': '#2ca02c',
        'SV-Grid': '#d62728', 'SV': '#9467bd', 'SVMR': '#8c564b'
    }
    
    # Plotar cada modelo
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['total_model_cost'],
                   model_data['total_model_latency'],
                   label=model,
                   marker=markers.get(model.split('_')[0], 'o'),
                   color=colors.get(model.split('_')[0], '#333333'),
                   alpha=0.8,
                   s=200,
                   edgecolors='white',
                   linewidth=1)
        
        # Adicionar anotações dos pesos com melhor posicionamento
        for _, row in model_data.iterrows():
            plt.annotate(f'w={row["cost_weight"]}',
                        (row['total_model_cost'], row['total_model_latency']),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=12,
                        bbox=dict(facecolor='white', 
                                edgecolor='gray',
                                alpha=0.9,
                                pad=3,
                                boxstyle='round,pad=0.5'),
                        arrowprops=dict(arrowstyle='->', 
                                      connectionstyle='arc3,rad=0.2',
                                      color='gray'))
    
    # Melhorar a aparência do gráfico
    plt.xlabel('Custo Total', fontsize=14, fontweight='bold')
    plt.ylabel('Latência Total (s)', fontsize=14, fontweight='bold')
    plt.title(f'Fronteira de Pareto - {pop_name}', 
             fontsize=16, 
             fontweight='bold', 
             pad=20)
    
    # Aumentar tamanho dos números nos eixos
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Melhorar a legenda
    plt.legend(bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0,
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=12)
    
    # Melhorar o grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Ajustar os limites dos eixos com margem
    x_min, x_max = df['total_model_cost'].min(), df['total_model_cost'].max()
    y_min, y_max = df['total_model_latency'].min(), df['total_model_latency'].max()
    
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # Ajustar o layout e salvar
    output_dir = "analysis_results/pareto_comparison"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    
    # Salvar em SVG com alta qualidade
    plt.savefig(os.path.join(output_dir, f'pareto_frontier_{pop_name}.svg'), 
                format='svg',
                bbox_inches='tight',
                dpi=300)
    
    # Salvar também em PNG para visualização rápida
    plt.savefig(os.path.join(output_dir, f'pareto_frontier_{pop_name}.png'),
                format='png',
                bbox_inches='tight',
                dpi=300)
    plt.close()

if __name__ == "__main__":
    # Plotar separadamente para cada população
    plot_pareto_frontier("Pop1")
    plot_pareto_frontier("Pop2")
    print("Gráficos das fronteiras de Pareto concluídos. Verifique a pasta analysis_results/pareto_comparison.")
