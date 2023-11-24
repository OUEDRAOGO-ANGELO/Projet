
import os
import dask.dataframe as dd
import pandas as pd

def lire_hdf_dask(nom_fichier, repertoire=os.getcwd()):
    """
    Charge un fichier HDF5 dans un DataFrame Dask.
    
    Si le fichier HDF5 n'est pas partitionnable et qu'il n'y a pas déjà une version partitionnée,
    une version partitionnée sera créée dans le répertoire 'data/data_dask'.
    
    Paramètres:
    - nom_fichier (str): Nom du fichier HDF5.
    - repertoire (str, facultatif): Répertoire contenant le fichier HDF5. Par défaut, 'data/data_extracted'.

    Retour:
    - ddf: DataFrame Dask
    """
    repertoire_dask = 'data/data_dask'
    
    # Vérifiez si le répertoire dask existe; sinon, créez-le
    if not os.path.exists(repertoire_dask):
        os.makedirs(repertoire_dask)
        
    fichier_partitionne = os.path.join(repertoire_dask, f"{nom_fichier[:-3]}_dask.h5")
    
    # Vérifiez si une version partitionnée du fichier existe
    if os.path.exists(fichier_partitionne):
        return dd.read_hdf(fichier_partitionne, '*')

    chemin_fichier = os.path.join(repertoire, nom_fichier)
    
    try:
        return dd.read_hdf(chemin_fichier, '*')
    except TypeError:
        with pd.HDFStore(chemin_fichier) as store:
            keys_store = store.keys()
            if not keys_store:
                raise ValueError(f"Aucun jeu de données trouvé dans HDFStore: {nom_fichier}")

            print("Ce HDFStore n'est pas partitionnable et ne peut être utilisé de manière monolithique qu'avec pandas.")
            print(f"Création d'un nouveau fichier de données: '{nom_fichier[:-3]}_dask.h5'")
            
            # Créez une version partitionnée du fichier
            with pd.HDFStore(fichier_partitionne, mode='w') as h:
                for key in keys_store:
                    h.put(key, store[key], format='table')
            print(f"Lecture du fichier de données: '{nom_fichier[:-3]}_dask.h5'")
            return dd.read_hdf(fichier_partitionne, '*')
