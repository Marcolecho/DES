import numpy as np
from tensorflow import keras
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

# --- 1. REPRISE DES FONCTIONS DE CONFIGURATION DE VOTRE CODE ---

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

"""
cation = "C=C/C=C/Cn1cc[n+](C)c1"
anion = "CCCC[C@H](C)[O-]"
crystal = "MCC"
Ts = 80
heating_time = 5
"""


nouveau_cation = "C=C/C=C/Cn1cc[n+](C)c1"
nouvel_anion = "CCCC[C@H](C)[O-]"         
nouvelle_T = 200            
nouveau_temps = 5                
nouveau_type_cellulose = "MCC"      

# --- 3. ENCODAGE DE LA NOUVELLE ENTRÉE ---

# a) Calcul des descripteurs moléculaires et fingerprints
fp_cat = smiles_to_MACCS(nouveau_cation)
des_cat = getMolDescriptors(nouveau_cation)

fp_ani = smiles_to_MACCS(nouvel_anion)
des_ani = getMolDescriptors(nouvel_anion)

# b) One-Hot Encoding manuel du type de cellulose (comme dans votre boucle)
c_avicel = 1 if nouveau_type_cellulose == 'Avicel' else 0
c_mcc = 1 if nouveau_type_cellulose == 'MCC' else 0
c_cellulose = 1 if nouveau_type_cellulose == 'cellulose' else 0

# c) Assemblage de toutes les caractéristiques en une seule ligne (Vecteur X_new)
# On utilise np.array([ ... ]) pour créer une matrice à 2 dimensions (1 ligne, 759 colonnes)
X_new = np.c_[
    [fp_cat], [des_cat], 
    [fp_ani], [des_ani], 
    [nouvelle_T], [nouveau_temps], 
    [c_avicel], [c_mcc], [c_cellulose]
]


model = keras.models.load_model('model_cellulose.keras')
# Faire la prédiction
prediction = model.predict(X_new)

# Afficher le résultat
print("\n" + "="*40)
print(f"Propriétés de l'essai : T={nouvelle_T}°C, Temps={nouveau_temps}h, Cellulose={nouveau_type_cellulose}")
print(f"Résultat de la prédiction (Solubilité estimée) : {prediction[0][0]:.4f}")
print("="*40)