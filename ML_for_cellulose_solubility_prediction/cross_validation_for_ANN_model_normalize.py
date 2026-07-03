import joblib
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def getMolDescriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    res = []
    for nm, fn in Descriptors._descList:
        res.append(fn(mol))
    return res


def smiles_to_MACCS(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return np.array(fp, float)


if __name__ == '__main__':

    df = pd.read_csv(
        'dataset_for_cellulose_solubility_ML_model_water_content_less_1%.csv'
    )

    smis = df['smiles'].tolist()
    cation = df['cation'].tolist()
    anion = df['anion'].tolist()
    Ts = df['T'].tolist()
    heating_time = df['heating_time'].tolist()
    Ys = df['solv'].values.reshape(-1, 1)

    Crystal = df['cellulose_crystal'].tolist()

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
        else:
            Crystal_avicel.append(0)
            Crystal_MCC.append(0)
            Crystal_cellulose.append(1)

    print("Calcul des descripteurs et empreintes moléculaires...")
    Des_cation = np.array([getMolDescriptors(c) for c in cation])
    Des_anion = np.array([getMolDescriptors(a) for a in anion])

    fp_cation = np.array([smiles_to_MACCS(c) for c in cation])
    fp_anion = np.array([smiles_to_MACCS(a) for a in anion])

    X_fp = np.c_[fp_cation, fp_anion]
    X_desc = np.c_[Des_cation, Des_anion]
    X_cont = np.c_[Ts, heating_time]
    X_cat = np.c_[Crystal_avicel, Crystal_MCC, Crystal_cellulose]

    # Ajustement et transformation des scalers sur l'ensemble de données (100%)
    scaler_desc = StandardScaler()
    scaler_cont = StandardScaler()

    X_desc = scaler_desc.fit_transform(X_desc)
    X_cont = scaler_cont.fit_transform(X_cont)

    X_final = np.c_[X_fp, X_desc, X_cont, X_cat]
    
    kf = KFold(n_splits=10, shuffle=True, random_state=1)

    pred = np.zeros_like(Ys)

    keras.utils.set_random_seed(1)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print("Début de la Cross-Validation (10 plis)...")

    for k, (train_index, test_index) in enumerate(kf.split(X_final), 1):

        X_train = X_final[train_index]
        X_test = X_final[test_index]

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(759, input_shape=(X_final.shape[1],), activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(759, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))

        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(0.0001)
        )

        model.fit(
            X_train,
            Ys[train_index],
            epochs=7000,
            batch_size=64,
            verbose=0
        )

        pred[test_index] = model.predict(X_test, verbose=0)
        print(f"Pli {k}/10 calculé.")

    print("\n--- PERFORMANCES HORS-PILOTE (ÉVALUATION) ---")
    print("R2: ", r2_score(Ys, pred))
    print("MSE: ", mean_squared_error(Ys, pred))

    print("\nLancement de l'entraînement final sur 100% du jeu de données...")
    
    model_production = keras.models.Sequential()
    model_production.add(keras.layers.Dense(759, input_shape=(X_final.shape[1],), activation='relu'))
    model_production.add(keras.layers.Dropout(0.5))
    model_production.add(keras.layers.Dense(759, activation='relu'))
    model_production.add(keras.layers.Dropout(0.5))
    model_production.add(keras.layers.Dense(1))

    model_production.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(0.0001)
    )

    model_production.fit(
        X_final,
        Ys,
        epochs=7000,
        batch_size=64,
        verbose=1  
    )

    print("\nSauvegarde des fichiers...")
    model_production.save('model_cellulose_normalize_v3.keras')
    joblib.dump(scaler_desc, 'scaler_desc_v3.pkl')
    joblib.dump(scaler_cont, 'scaler_cont_v3.pkl')
    
    print("Modèle final et Scalers sauvegardés avec succès !")