# DES
Générateur de DES à partir d'un base de données de DES et des modèles IsolationForest et KNN

## Architecture

On retrouve à la racine 3 fichiers python : 
* **des_train.ipynb :** permet la préparation des données
    * Le nettoyage
    * La transformation
    * L'entrainement des 2 modèles

* **if_calc.py :** fichier à lancer après la fin de **des_train.ipynb**. Il permet de faire le trie entre les potentiel et les non potentiel DES

* **knn_calc.py :** fichier à lancer après **id_calc.py**. Il permet de ne garder les potentiels DES proches chimiquement d'autres vrais DES

<br><br>

Le dossier **results** contient la sortie des 3 fichiers python :  
* **df_AllSources.csv :** Sources des différents articles donnant des données de DES pour notre projet

* **df_Final_Ia.csv :** Dataframe d'entrainement entièrement traité et prêt à être utilisé pour l'IA

* **DES_probab.csv :** Fichier de sortie apres IsolationForest de **if_calc.py**

* **DES_more_probab.csv :** Fichier de sortie apres KNN de **knn_calc.py**

<br><br>

Le dossier **saveModel** contient des modèles entrainés et outils à utiliser pour faire de la prédictions pour **if_calc.py** et **knn_calc.py**

<br><br>

les dossiers **DES_DATA** et **HB_DATA** stockent les différentes données récupérés de pré-traitement 

<br><br>

# Sources des données de base

- https://link.springer.com/article/10.1186/s13321-019-0381-4#Sec2

- https://doi.org/10.1021/acssuschemeng.3c05207          
    * https://github.com/AstyLavrinenko/Eutectic-prediction/tree/main    
        * DES_init_uptate

- ACID Design / Lab SCAMT Institute (ITMO University)   
    * https://github.com/acid-design-lab/DESignSolvents/tree/main
        * Density
        * Melting_temperature
        * Viscosity
