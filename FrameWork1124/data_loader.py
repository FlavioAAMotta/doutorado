import os
import pandas as pd
import yaml
import logging

class DataLoader:
    def __init__(self, data_dir='data/', config_path='config/config.yaml'):
        self.data_dir = data_dir
        self.params = self.load_params(config_path)
        logging.info("Parâmetros carregados com sucesso.")

    def load_access_data(self, pop_name):
        file_name = f'access_{pop_name}.txt'
        path = os.path.join(self.data_dir, file_name)
        try:
            df = pd.read_csv(path, sep=' ')
            logging.info(f"Dados de acesso carregados de {file_name}.")
            return df
        except FileNotFoundError as e:
            logging.error(f"Arquivo não encontrado: {e}")
            raise

    def load_volume_data(self, pop_name):
        file_name = f'vol_bytes_{pop_name}.txt'
        path = os.path.join(self.data_dir, file_name)
        try:
            df = pd.read_csv(path, sep=' ')
            logging.info(f"Dados de volume carregados de {file_name}.")
            return df
        except FileNotFoundError as e:
            logging.error(f"Arquivo não encontrado: {e}")
            raise

    def load_params(self, config_path):
        try:
            with open(config_path, 'r') as file:
                params = yaml.safe_load(file)
            logging.info(f"Parâmetros carregados de {config_path}.")
            return params
        except FileNotFoundError as e:
            logging.error(f"Arquivo de configuração não encontrado: {e}")
            raise
