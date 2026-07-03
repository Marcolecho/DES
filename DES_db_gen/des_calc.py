from rdkit import Chem
import csv
import pandas as pd
import requests
from urllib.request import urlopen
from urllib.parse import quote
import pubchempy as pcp
from math import nan
import numpy as np
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def extract_data_to_csv(sdf_path, filename):
    dataHB = []

    suppl = Chem.SDMolSupplier(sdf_path, sanitize=True)

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["SMILES", "Masse_Molaire"])

        for mol in suppl:
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol)

            if mol.HasProp("Molecular_Weight"):
                mw = mol.GetProp("Molecular_Weight")
            else:
                from rdkit.Chem import Descriptors
                mw = Descriptors.MolWt(mol)

            dataHB.append((smiles, mw))
            writer.writerow([smiles, mw])
        
    return dataHB

def returnOneHotEncod(type):
    if(type == "I" or type == 1 or type == "1"):   return [0,1,0,0,0,0]
    if(type == "II" or type == 2 or type == "2"):  return [0,0,1,0,0,0]
    if(type == "III" or type == 3 or type == "3"): return [0,0,0,1,0,0]
    if(type == "IV" or type == 4 or type == "4"):  return [0,0,0,0,1,0]
    if(type == "V" or type == 5 or type == "5"):   return [0,0,0,0,0,1]
    return [1,0,0,0,0,0]

def evaluer_nouveau_couple(famille, smi_hba, smi_hbd, ratio_x1, ratio_x2, scaler_rdkit, model_ia):

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
    
    vecteur_test = np.hstack((familleOneHot, ratio_x1, ratio_x2, rdkit_1_scaled, rdkit_2_scaled, maccs_1, maccs_2))
    
    pred = model_ia.predict([vecteur_test])
    score = model_ia.score_samples([vecteur_test]) 
    
    resScore = round(abs(score[0]),5)

    if pred[0] == 1:
        return {"result": True, "score": resScore}
    else:
        return {"result": False, "score": resScore}
    

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


scaler = joblib.load('scaler.pkl')
model_iforest = joblib.load('isolation_forest_des.pkl')

smiles_list_HBA = extract_data_to_csv("./HB_DATA/13321_2019_381_MOESM2_ESM.sdf", "./HB_DATA/HBA.csv")
smiles_list_HBD = extract_data_to_csv("./HB_DATA/13321_2019_381_MOESM3_ESM.sdf", "./HB_DATA/HBD.csv")

ratioA = 0.1
ratioB = 0.9
total = ratioA + ratioB
nbNoDES = 0

with open('DES_probab.csv', "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["HBA", "HBD", "score", "ratioHBA", "ratioHBD"])

for hba_mol in smiles_list_HBA:
    hbaTest = hba_mol[0]
    for hbd_mol in smiles_list_HBD:
        hbdTest = hbd_mol[0]

        bestResponse = {"bestScore": 1, "ratioA": 0, "ratioB": 0}

        for i in np.arange(ratioA, ratioB+ratioA, 0.1):
            r1 = round(i,2)
            r2 = round(total-i,2)
            response = evaluer_nouveau_couple(determiner_type_des(hbaTest, hbdTest), hbaTest, hbdTest, r1, r2, scaler, model_iforest)
            if(response['result'] == True):
                if(response['score'] < bestResponse['bestScore']):
                    bestResponse['bestScore'] = response['score']
                    bestResponse['ratioA'] = r1
                    bestResponse['ratioB'] = r2


        if(bestResponse['bestScore'] != 1): 
            print(f"best score for {hbaTest} and {hbdTest} - {bestResponse['bestScore']} - r1:{bestResponse['ratioA']} r2:{bestResponse['ratioB']}")

            with open('DES_probab.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([hbaTest, hbdTest, bestResponse['bestScore'], bestResponse['ratioA'], bestResponse['ratioB']])
        else:
            nbNoDES += 1
            print(f"no DES", nbNoDES)