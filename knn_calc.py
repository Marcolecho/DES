import csv
from rdkit import Chem
import joblib
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Descriptors

"""
Converti la colonne Type de DES en plusieurs listes où une catégorie de la colonne type de DES deviens une colonne à part entière
C'est ce que l'on appelle du OneHotEncoding
"""
def returnOneHotEncod(type):
    if(type == "I" or type == 1 or type == "1"):   return [0,1,0,0,0,0]
    if(type == "II" or type == 2 or type == "2"):  return [0,0,1,0,0,0]
    if(type == "III" or type == 3 or type == "3"): return [0,0,0,1,0,0]
    if(type == "IV" or type == 4 or type == "4"):  return [0,0,0,0,1,0]
    if(type == "V" or type == 5 or type == "5"):   return [0,0,0,0,0,1]
    return [1,0,0,0,0,0]


"""
Fonction qui pour un SMILES de HBA et HBD renvoie un nombre entre 1 et 5 correspondant à la famille de DES auquel le potentiel DES appartient
en fonction des caractèristique chimique

fonction crée pas chatGPT, j'ai pas les compétences au niveau chimie
"""
def determiner_type_des(smi_hba, smi_hbd):
    """
    Détermine automatiquement le Type de DES (I, II, III, IV, V) 
    à partir des structures SMILES du HBA et du HBD.
    """
    # 1. Détection du Type V (Pas d'ions du tout)
    # Si aucun des deux smiles ne contient de charge '+' ou '-' ou d'atome métallique isolé
    if "+" not in smi_hba and "-" not in smi_hba and "+" not in smi_hbd and "-" not in smi_hbd:
        # Vérification s'il y a un métal (ex: Zn, Fe, Cr) sous forme neutre (rare pour les types I-IV)
        metaux = ["Zn", "Fe", "Cr", "Al", "Cu"]
        if not any(m in smi_hba or m in smi_hbd for m in metaux):
            return 5

    # 2. Détection du Type IV (Le HBA est un métal/sel métallique, le HBD est organique)
    # Souvent caractérisé par l'absence d'azote quaternaire [N+] ou phosphonium [P+] dans le HBA
    if "[N+]" not in smi_hba and "[P+]" not in smi_hba and ("Zn" in smi_hba or "Al" in smi_hba):
        return 4

    # 3. Pour les Types I, II, III : Le HBA est un sel organique quaternaire (ex: Choline)
    # On regarde la nature du HBD
    
    # Cas du Type II : Présence d'eau de cristallisation (Hydrate) dans le HBD
    if "O" in smi_hbd and ("Cl" in smi_hbd or "[Cl-]" in smi_hbd) and any(m in smi_hbd for m in ["Cr", "Fe", "Al", "Cu"]):
        if "O" in smi_hbd: # Si l'expression contient de l'eau (ex: .O.O.O.O.O.O)
            return 2
        return 1
        
    # Cas du Type I : Sel métallique anhydre
    if any(m in smi_hbd for m in ["Zn", "Fe", "Al", "Sn"]) and "O" not in smi_hbd:
        return 1

    # Cas du Type III : Le HBD est organique et neutre (contient C, H, O, N et pas de métaux)
    return 3

"""
Pour une famille, SMILES HBA/HBD et ratio, on construit le vecteur ayant le format accepté pour prédire 
"""
def returnVectorTry(famille, smi_hba, smi_hbd, ratio_x1, ratio_x2, scaler_rdkit):

    mol1 = Chem.MolFromSmiles(smi_hba)
    mol2 = Chem.MolFromSmiles(smi_hbd)
    
    if mol1 is None or mol2 is None:
        return "Erreur : Un des SMILES est chimiquement invalide."
        
    familleOneHot = returnOneHotEncod(famille)
        
    rdkit_1 = [fn(mol1) for nm, fn in Descriptors._descList]
    rdkit_2 = [fn(mol2) for nm, fn in Descriptors._descList]
    
    rdkit_1 = np.nan_to_num(rdkit_1, nan=0.0, posinf=0.0, neginf=0.0)
    rdkit_2 = np.nan_to_num(rdkit_2, nan=0.0, posinf=0.0, neginf=0.0)
    
    rdkit_1_scaled = scaler_rdkit.transform([rdkit_1])[0]
    rdkit_2_scaled = scaler_rdkit.transform([rdkit_2])[0]
    
    maccs_1 = list(AllChem.GetMACCSKeysFingerprint(mol1))
    maccs_2 = list(AllChem.GetMACCSKeysFingerprint(mol2))
    
    return np.hstack((familleOneHot, ratio_x1, ratio_x2, rdkit_1_scaled, rdkit_2_scaled, maccs_1, maccs_2))

"""
Pour la liste de DES valide sauvegarde les résultats dans le fichier prévu à cet effet 
"""
def saveToFile(nameFile, df_DES_valides, DES_propab):
    print(f"Nombre de solvants retenus : {len(df_DES_valides)} sur {len(DES_propab)}")

    with open(nameFile, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["HBA", "HBD", "score", "ratioHBA", "ratioHBD"])

    with open(nameFile, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in df_DES_valides.values:
            writer.writerow([i[0], i[1], i[6], i[2], i[3]])

if __name__ == '__main__':
    knn_flash =  joblib.load('knn.pkl')
    scaler = joblib.load('scaler.pkl')
    model_iforest = joblib.load('isolation_forest_des.pkl')
    DES_propab = pd.read_csv('./DES_probab.csv')

    limitEcartType = 5
    limitMoyenne = 15
    nameFile = './DES_more_probab.csv'

    lignes_validees = []

    """ 
    Pour chaque ligne dans la liste de DES ayant passé le test de l'IsolationForest
        On récupère le vecteur à partir du des 'i' à tester que l'on passe sur dans le modèle knn
        On récupère les 3 points les plus proches avec leurs distances par rapport au point 'i'
        on récupère l'écart type et la moyenne de la distance
        si ces deux valeurs sont sous les limites donné (limitEcartType et limitMoyenne)
            on ajoute 'i' à la une nouvelle liste des DES ayant passé isolation forest et knn
        on lance la fonction pour sauvegarder cette liste  
    """
    for index, i in enumerate(DES_propab.values):
        hba = i[0]
        hbd = i[1]
        score_ia_candidat = i[2]  
        ratioHBA = i[3]
        ratioHBD = i[4]

        vectorDES = returnVectorTry(determiner_type_des(hba, hbd), hba, hbd, ratioHBA, ratioHBD, scaler)
        distances, indices_voisins = knn_flash.kneighbors([vectorDES])

        distances_voisins = distances[0]
        
        distance_ecartType = np.std(distances_voisins)
        distance_moyenne = np.mean(distances_voisins)
        
        if distance_ecartType < limitEcartType:
            if distance_moyenne < limitMoyenne:
                lignes_validees.append({
                    'HBA': hba,
                    'HBD': hbd,
                    'ratioHBA': ratioHBA,
                    'ratioHBD': ratioHBD,
                    'distance_moyenne': distance_moyenne,
                    'distance_std': distance_ecartType,
                    'score_initial_candidat': score_ia_candidat
                })

    saveToFile(nameFile, pd.DataFrame(lignes_validees), DES_propab)