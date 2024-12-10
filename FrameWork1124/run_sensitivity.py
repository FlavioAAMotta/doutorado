import os
import numpy as np
import pandas as pd
from main import *
from models.user_profiles import UserProfile
from data_loader import DataLoader
from main import get_time_windows, run_analysis  # Explicitly import needed functions
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

def confusion_matrix_to_labels(conf_matrix):
    """
    Convert a confusion matrix to true labels and predicted labels.
    
    Parameters:
    conf_matrix : 2D array
        The confusion matrix (2x2 for binary classification)
        Format: [[TN, FP], [FN, TP]]
    
    Returns:
    tuple : (y_true, y_pred) containing the true and predicted labels
    """
    TN, FP, FN, TP = conf_matrix.ravel()
    
    # Create true labels
    y_true = [1] * (TP + FN) + [0] * (TN + FP)
    
    # Create predicted labels
    y_pred = [1] * TP + [0] * FN + [1] * FP + [0] * TN
    
    return y_true, y_pred

def run_multiple_weights():
    # Load configuration and models
    data_loader = DataLoader()
    models_to_run = data_loader.params['models_to_run']
    window_size = data_loader.params['window_size']
    step_size = data_loader.params['step_size']
    pop_name = data_loader.params['pop_name']
    
    # Load data and create windows
    df_access = data_loader.load_access_data(pop_name)
    windows = get_time_windows(df_access, window_size, step_size)
    
    # Define range of weights to test
    weights = [1] + list(range(10, 91, 10)) + [99]
    
    # Create sensitivity results directory
    sensitivity_dir = os.path.join('results', 'sensitivity_analysis')
    os.makedirs(sensitivity_dir, exist_ok=True)
    
    # Store results for all weights
    all_results = []
    
    # Run for each weight
    for weight in weights:
        print(f"\nRunning with cost_weight = {weight}")
        
        # Create new UserProfile with current weight
        user_profile = UserProfile(weight)
        
        # Initialize results for this weight
        cumulative_results = {}
        for model_name in models_to_run:
            cumulative_results[model_name] = {
                'model_cost': [],
                'oracle_cost': [],
                'model_latency': [],
                'oracle_latency': [],
                'confusion_matrix': np.zeros((2, 2), dtype=int)
            }
        
        # Initialize final_results
        final_results = {model_name: {} for model_name in models_to_run}
        
        # Run analysis for each window
        for window in windows:
            # Aqui você precisa adicionar o código que:
            # 1. Carrega os dados para esta janela
            results = run_analysis(window, user_profile)
            
            # 2. Atualiza cumulative_results com os resultados desta janela
            for model_name, model_results in results.items():
                cumulative_results[model_name]['model_cost'].append(model_results['cost'])
                cumulative_results[model_name]['oracle_cost'].append(model_results['oracle_cost'])
                cumulative_results[model_name]['model_latency'].append(model_results['latency'])
                cumulative_results[model_name]['oracle_latency'].append(model_results['oracle_latency'])
                cumulative_results[model_name]['confusion_matrix'] += model_results['confusion_matrix']
        
        # Calculate final metrics for this weight
        for model_name in models_to_run:
            conf_matrix = cumulative_results[model_name]['confusion_matrix']
            y_true, y_pred = confusion_matrix_to_labels(conf_matrix)
            
            final_results[model_name] = {
                'model': model_name,
                'cost_weight': weight,
                'total_model_cost': np.sum(cumulative_results[model_name]['model_cost']),
                'total_oracle_cost': np.sum(cumulative_results[model_name]['oracle_cost']),
                'total_model_latency': np.sum(cumulative_results[model_name]['model_latency']),
                'total_oracle_latency': np.sum(cumulative_results[model_name]['oracle_latency']),
                'confusion_matrix': conf_matrix.tolist(),
                'accuracy': np.trace(conf_matrix) / np.sum(conf_matrix),
                'f1_score': f1_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_pred)
            }
            all_results.append(final_results[model_name])
    
    # Convert all results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Convert DataFrame columns to numpy arrays before saving
    results_df['cost_weight'] = results_df['cost_weight'].to_numpy()
    
    # Save combined results
    os.makedirs(sensitivity_dir, exist_ok=True)
    results_df.to_csv(os.path.join(sensitivity_dir, 'all_weights_results.csv'), index=False)

if __name__ == "__main__":
    run_multiple_weights() 