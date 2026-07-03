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


def getMolDescriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    res = []
    for nm, fn in Descriptors._descList:
        res.append(fn(mol))
    return res

def smiles_to_MACCS(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    fp = np.array(fp, float)
    return fp

if __name__ == '__main__':
    df = pd.read_csv('dataset_for_cellulose_solubility_ML_model_water_content_less_1%.csv')
    smis = list(df.loc[:, 'smiles'])
    cation = list(df.loc[:, 'cation'])
    anion = list(df.loc[:, 'anion'])
    Ts = list(df.loc[:, 'T'])
    Crystal = list(df.loc[:, 'cellulose_crystal'])
    Crystal_avicel = []
    Crystal_MCC = []
    Crystal_cellulose = []
    for item in Crystal:
        if item == 'Avicel':
            Crystal_avicel.append(1)
            Crystal_MCC.append(0)
            Crystal_cellulose.append(0)
        
        elif item == 'MCC':
            Crystal_avicel.append(0)
            Crystal_MCC.append(1)
            Crystal_cellulose.append(0)
            
        elif item == 'cellulose':
            Crystal_avicel.append(0)
            Crystal_MCC.append(0)
            Crystal_cellulose.append(1)

    heating_time = list(df.loc[:,'heating_time'])
    Ys = list(df.loc[:, 'solv'])

    Des_cation = np.array([getMolDescriptors(c) for c in cation])
    Des_anion = np.array([getMolDescriptors(a) for a in anion])
    fp_anion = np.array([smiles_to_MACCS(a) for a in anion])
    fp_cation = np.array([smiles_to_MACCS(c) for c in cation])
    fp_mol = np.array([smiles_to_MACCS(s) for s in smis])
    X = np.c_[fp_cation, Des_cation, fp_anion, Des_anion, Ts, heating_time, Crystal_avicel, Crystal_MCC, Crystal_cellulose]
    Y = np.array(Ys).reshape(len(Ys), 1)

    kf = KFold(n_splits=10)
    pred = np.zeros(len(Y)).reshape(len(Y), 1)
    k = 0
    regr_set = []

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    keras.utils.set_random_seed(1)
    for train_index, test_index in kf.split(X, Y):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=759, input_shape=(None, 759), activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(units=759, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(units=1))
        model.compile(loss='MSE', optimizer=tf.keras.optimizers.Adam(0.0001))
        model.fit(X[train_index], Y[train_index], epochs=7000)
        pred[test_index] = model.predict(X[test_index])
        k = k + 1
        print(k)

    print("R2: ", r2_score(Y, pred))   
    print("MSE: ", mean_squared_error(Y, pred))

    model.save('model_cellulose.keras')
    print("Modèle sauvegardé avec succès dans le fichier 'model_cellulose.keras' !")


