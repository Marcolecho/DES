import csv
import numpy as np
from tensorflow import keras
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd

# --- PARAMÈTRES PHYSIQUES CIBLÉS (3-4 VARIATIONS) ---
LISTE_TEMP_CIBLES = [40, 80, 120, 160, 200]       # Variations de température
LISTE_TEMPS_CIBLES = [0.5, 2.0, 5.0, 10.0]     # Variations de temps de chauffe
LISTE_CELLULOSE_CRYSTALS = ['Avicel', 'MCC', 'cellulose']

print("Chargement du modèle...")
# On charge uniquement le modèle
model = keras.models.load_model('model_cellulose.keras')

def getMolDescriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: 
        return [0.0] * len(Descriptors._descList)
    return [fn(mol) for nm, fn in Descriptors._descList]

def smiles_to_MACCS(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: 
        return np.zeros(167, float) 
    return np.array(AllChem.GetMACCSKeysFingerprint(mol), float)


# --- FONCTION DE PRÉDICTION ADAPTÉE (SANS SCALER) ---
def calcBestSol(cation, anion, scenarios_physiques, metadonnees):
    fp_cat = smiles_to_MACCS(cation)
    des_cat = getMolDescriptors(cation)
    fp_ani = smiles_to_MACCS(anion)
    des_ani = getMolDescriptors(anion)

    # Fusion des descripteurs bruts en vecteur 1D direct
    X_desc_raw = np.concatenate([des_cat, des_ani])

    # Assemblage de la partie chimie fixe brute [FP, DESC_RAW]
    chimie_fixe = np.concatenate([fp_cat, fp_ani, X_desc_raw])

    # Répétition de la chimie pour correspondre au nombre de scénarios physiques
    chimie_repete = np.tile(chimie_fixe, (len(scenarios_physiques), 1))
    X_giant = np.hstack([chimie_repete, scenarios_physiques])

    # Prédiction brute
    predictions = model.predict(X_giant, verbose=0) 

    # Recherche de la meilleure combinaison physique/cellulose
    index_meilleur = np.argmax(predictions) 
    meilleure_solubilite = predictions[index_meilleur][0]
    
    temp_opt, temps_opt, cellulose_opt = metadonnees[index_meilleur]

    if meilleure_solubilite < 0:
        meilleure_solubilite = 0.0

    return {
        'meilleure_solubilite': meilleure_solubilite,
        'temp_opt': temp_opt,
        'temps_opt': temps_opt,
        'cellulose_opt': cellulose_opt
    }


# --- PRÉPARATION DES SCÉNARIOS CIBLÉS (BRUTS) ---
scenarios_physiques = []
metadonnees = [] 

print("Génération de la matrice des variations cibles brutes...")

for cellulose_crystal in LISTE_CELLULOSE_CRYSTALS:
    c_avicel = 1.0 if cellulose_crystal == 'Avicel' else 0.0
    c_mcc = 1.0 if cellulose_crystal == 'MCC' else 0.0
    c_cellulose = 1.0 if cellulose_crystal == 'cellulose' else 0.0
    
    for temp in LISTE_TEMP_CIBLES:
        for heating_time in LISTE_TEMPS_CIBLES:
            # On stocke directement les valeurs brutes (temp, heating_time) sans passer par un transform()
            scenarios_physiques.append([float(temp), float(heating_time), c_avicel, c_mcc, c_cellulose])
            
            # Métadonnées pour le fichier final
            metadonnees.append((temp, heating_time, cellulose_crystal))

scenarios_physiques = np.array(scenarios_physiques)
print(f"Grille brute prête ! {len(scenarios_physiques)} scénarios par couple moléculaire.")


# --- SCREENING ET ÉCRITURE EN TEMPS RÉEL ---
df_cations = pd.read_csv('pubchem_cations.csv')
df_anions = pd.read_csv('pubchem_anions.csv')

with open('results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Cation', 'Anion', 'Meilleure Solubilite', 'Temp Opt', 'Temps Opt', 'Cellulose Opt'])

# Boucle de screening moléculaire
for cation in df_cations['smiles'].values:
    for anion in df_anions['smiles'].values:
        print(f"Calcul en cours -> Cation: {cation} | Anion: {anion}")
        
        bestScore = calcBestSol(cation, anion, scenarios_physiques, metadonnees)
        
        # Enregistrement si la solubilité estimée est supérieure à 5
        if bestScore['meilleure_solubilite'] > 5:
            with open('results.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    cation, 
                    anion, 
                    f"{bestScore['meilleure_solubilite']:.4f}",
                    bestScore['temp_opt'],
                    bestScore['temps_opt'],
                    bestScore['cellulose_opt']
                ])

print("\nScreening des variations terminé ! Les résultats optimaux sont dans 'results.csv'.")