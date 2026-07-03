

from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


df = pd.read_csv('dataset_for_cellulose_solubility_ML_model_water_content_less_1%.csv')

temperatures = df['T'].values
heating_times = df['heating_time'].values
cellulose_crystals = df['cellulose_crystal'].values

min_temp = min(temperatures)
max_temp = max(temperatures)
min_heating_time = min(heating_times)
max_heating_time = max(heating_times)
type_cellulose_crystals = set(cellulose_crystals)

for cellulose_crystal in type_cellulose_crystals:
    for temp in range(min_temp, max_temp + 1):
        for heating_time in np.arange(min_heating_time, max_heating_time + 0.1, 0.1):
            print(temp, round(heating_time, 1), cellulose_crystal)
