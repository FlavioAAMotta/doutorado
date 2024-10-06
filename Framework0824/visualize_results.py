# visualize_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar o estilo dos gráficos
sns.set(style='whitegrid')

# Carregar resultados do CSV
results = pd.read_csv('results.csv')

# Exibir as primeiras linhas do DataFrame
print(results.head())

# Verificar se a coluna 'window' existe
if 'window' not in results.columns:
    print("A coluna 'window' não está presente no DataFrame.")
    # Se 'window' não existir, podemos tentar inferir ou alertar o usuário
    # Para este caso, interromperemos o script
    raise KeyError("'window' column is missing in the CSV file.")

# Converter a coluna 'window' para tipo inteiro, se necessário
results['window'] = results['window'].astype(int)

# Visualizar resultados estatísticos gerais
print("\nEstatísticas Descritivas:")
print(results.describe())

# Separar os resultados por modelo
models = results['Model'].unique()

# Plotar RCS (Relative Cost Savings) para cada modelo ao longo das janelas
plt.figure(figsize=(12, 6))
for model in models:
    model_data = results[results['Model'] == model]
    plt.plot(model_data['window'], model_data['rcs ml'], label=model)

plt.xlabel('Janela (Semana)')
plt.ylabel('RCS (Relative Cost Savings)')
plt.title('RCS ao longo do tempo por Modelo')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rcs_over_time.png')
plt.show()

# Plotar Accuracy para cada modelo ao longo das janelas
plt.figure(figsize=(12, 6))
for model in models:
    model_data = results[results['Model'] == model]
    plt.plot(model_data['window'], model_data['Accuracy'], label=model)

plt.xlabel('Janela (Semana)')
plt.ylabel('Acurácia')
plt.title('Acurácia ao longo do tempo por Modelo')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_over_time.png')
plt.show()

# Plotar custos totais para cada modelo
plt.figure(figsize=(10, 6))
model_costs = results.groupby('Model')['cost ml'].mean().reset_index()
sns.barplot(x='Model', y='cost ml', data=model_costs)
plt.xlabel('Modelo')
plt.ylabel('Custo Médio')
plt.title('Custo Médio por Modelo')
plt.tight_layout()
plt.savefig('average_cost_by_model.png')
plt.show()

# Análise estatística
print("\nAnálise Estatística:")
for metric in ['Accuracy', 'Precision', 'Recall', 'f1_score', 'auc_roc', 'rcs ml']:
    print(f"\nMétricas para {metric}:")
    metric_data = results.pivot_table(values=metric, index='window', columns='Model')
    print(metric_data.describe())
