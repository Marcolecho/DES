import joblib
import numpy as np
from tensorflow import keras
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

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


print("Chargement du modèle et des scalers...")
model = keras.models.load_model('model_cellulose_normalize_v3.keras')
scaler_desc = joblib.load('scaler_desc_v3.pkl')
scaler_cont = joblib.load('scaler_cont_v3.pkl')

nouveau_cation = "C=C/C=C/Cn1cc[n+](C)c1"
nouvel_anion = "CCCC[C@H](C)[O-]"         
nouvelle_T = 200            
nouveau_temps = 0.5                
nouveau_type_cellulose = "MCC"     

fp_cat = smiles_to_MACCS(nouveau_cation)
fp_ani = smiles_to_MACCS(nouvel_anion)
X_fp = np.concatenate([fp_cat, fp_ani]).reshape(1, -1)

des_cat = getMolDescriptors(nouveau_cation)
des_ani = getMolDescriptors(nouvel_anion)
X_desc_raw = np.concatenate([des_cat, des_ani]).reshape(1, -1)
X_desc_scaled = scaler_desc.transform(X_desc_raw)  

X_cont_raw = np.array([[float(nouvelle_T), float(nouveau_temps)]])
X_cont_scaled = scaler_cont.transform(X_cont_raw)  

c_avicel = 1.0 if nouveau_type_cellulose == 'Avicel' else 0.0
c_mcc = 1.0 if nouveau_type_cellulose == 'MCC' else 0.0
c_cellulose = 1.0 if nouveau_type_cellulose == 'cellulose' else 0.0
X_cat = np.array([[c_avicel, c_mcc, c_cellulose]])

X_new_final = np.c_[X_fp, X_desc_scaled, X_cont_scaled, X_cat]

print("Calcul de la prédiction...")
prediction = model.predict(X_new_final, verbose=0)

print("\n" + "="*40)
print(f"Propriétés de l'essai : T={nouvelle_T}°C, Temps={nouveau_temps}h, Cellulose={nouveau_type_cellulose}")
print(f"Résultat de la prédiction (Solubilité estimée) : {prediction[0][0]:.4f}")
print("="*40)