# Load access data from the data folder
import pandas as pd
import os

# Load access data from the data folder
def load_acc_data(pop):
    file_name = f'access_{pop}.txt'
    df = pd.read_csv(os.path.join('data', file_name), sep=' ')
    return df

def load_vol_data(pop):
    file_name = f'vol_bytes_{pop}.txt'
    df = pd.read_csv(os.path.join('data', file_name), sep=' ')
    return df

def load_params():
    # params is an object that contains all the parameters (window size, step size)
    params = {}
    # for now, we will hard code the parameters
    params['window_size'] = 1
    params['step_size'] = 1
    params['model'] = 'ONL'
    return params