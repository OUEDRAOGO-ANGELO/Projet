
import os
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import ipywidgets as widgets
from ipywidgets import interact, IntSlider
from .fonction_plot import nameunit, get_colname
import plotly.graph_objs as go
from plotly.subplots import make_subplots






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





def tracer_serie_temporelle(ddf, numero_partition, colonnes):
    """
    Trace la série temporelle pour les colonnes spécifiées à partir d'une table donnée dans le fichier HDF5.
    
    Paramètres:
    - ddf : dask dataframe
    - numero_partition: Le numéro de la table à charger à partir du fichier HDF5.
    - colonnes: Une liste de colonnes à tracer.
    """
    donnees = ddf.partitions[numero_partition].compute()
    
    # Tracer la série temporelle pour les colonnes spécifiées
    for col in colonnes:
        plt.figure(figsize=(12, 6))
        plt.plot(donnees[col], label=col, alpha=0.7)
        plt.title(f'Série temporelle de {col} - Vol: {numero_partition}')
        plt.xlabel('Temps')
        plt.ylabel(col)
        plt.grid(True)
        plt.legend()
        plt.show()
        
        
        
def graphe_comparaison_différence(ddf, numero_partition, abs, ord):
    """
    Trace la série temporelle pour les colonnes spécifiées à partir d'une table donnée dans le fichier HDF5.
    
    Paramètres:
    - ddf : dask dataframe
    - numero_partition: Le numéro de la table à charger à partir du fichier HDF5.
    - colonnes: Une liste de colonnes à tracer.
    """
    donnees = ddf.partitions[numero_partition].compute()
    
    # Tracer la série temporelle pour les colonnes spécifiées
    plt.figure(figsize=(12, 6))
    plt.plot(np.abs(donnees[abs]-donnees[ord]), alpha=0.7)
    plt.title(f'graphe de différence {abs} entre {ord} en fontion du temps - Vol: {numero_partition}')
    plt.xlabel('Temps')
    plt.ylabel('différence')
    plt.grid(True)
    plt.legend()
    plt.show()        
    
    
def moyenne_sur_colonne(ddf, col):
    """
    Trace la série temporelle pour les colonnes spécifiées à partir d'une table donnée dans le fichier HDF5.
    
    Paramètres:
    - ddf : dask dataframe
    - numero_partition: Le numéro de la table à charger à partir du fichier HDF5.
    - colonnes: Une liste de colonnes à tracer.
    """
    moyenne=[]
    """
    for numero_partition in range(ddf.npartitions):

        donnees = ddf.partitions[numero_partition].compute()
        moyenne.append(numpy.mean(donnees[col]))
    """
    moyenne=(ddf.map_partitions(lambda df: df[col].mean()).compute())
    
    # Tracer la série temporelle pour les colonnes spécifiées
    plt.figure(figsize=(12, 6))
    plt.plot(moyenne, alpha=0.7)
    plt.title(f'graphe de la moyenne de {col} en fonction des vols')
    plt.xlabel('vol')
    plt.ylabel(f'moyenne de {col}')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return moyenne    



def Regression_lineaire(ddf, numero_partition, abs, ord):
    """
    Trace la série temporelle pour les colonnes spécifiées à partir d'une table donnée dans le fichier HDF5.
    
    Paramètres:
    - ddf : dask dataframe
    - numero_partition: Le numéro de la table à charger à partir du fichier HDF5.
    - colonnes: Une liste de colonnes à tracer.
    """
    donnees = ddf.partitions[numero_partition].compute()
    
    # Ajoutez une colonne de 1 pour représenter la constante dans la régression
    X_with_intercept = sm.add_constant(donnees)

    # Créez un modèle de régression linéaire avec statsmodels
    model = sm.OLS(donnees[abs], donnees[ord])

    # Ajustez le modèle aux données
    results = model.fit()

    # Affichez les résultats incluant les p-values
    print(results.summary()) 
       

    print(donnees[abs].shape)
    
    
    # Tracer la série temporelle pour les colonnes spécifiées
    plt.figure(figsize=(12, 6))
    plt.scatter(donnees[abs],donnees[ord], alpha=0.7)
    plt.title(f'Evolution de  {abs} en fonction de {ord} au cours du Vol: {numero_partition}')
    plt.xlabel(f'{abs}')
    plt.ylabel(f'{ord}')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return results





class Affichage_Interactif_Error(ValueError):
    def __init__(self,filename,message):
        self.filename = filename
        self.message = message
        
    def __str__(self):
        return """Opset({self.filename})
    {self.message}""".format(self=self)
    
    
    
    



class  Affichage_Interactif:
    
    def __init__(self, storename, Nomfichier, ddf,  phase=None, pos=0, name=None, sortkey=None):
        self.Nomfichier=Nomfichier
        self.ddf=ddf
        

        self.records = ddf.map_partitions(lambda df: df.index.name).compute()

        nbmax = len(self.records)
        if (pos < 0) or (pos >= nbmax):
            pos = 0
        if nbmax>0:
            self.df = self.ddf.partitions[pos]
            colname = get_colname(self.df.columns,name)
            phase = get_colname(self.df.columns,phase,default=None)
        else:
            self.df = None
            colname = None
            phase = None
                
        self.sigpos = pos
        self.colname = colname
        self.phase = phase
        
        self._tmppos = 0 # Variable cachée pour l'itération.
    def make_figure(self,f,phase=None,pos=None,name=None):
        """ Crée l'affichage interactif des courbes.
        
            Cette fonction définit les différents éléments de l'affichage. 

            :param f:       un pointeur sur la figure à créer.
            :param phase:   le nom d'une colonne binaire contenant les
                            points à mettre en évidence si besoin.
            :param pos:     le numéro du signal à afficher en premier sinon
                            le premier signal du fichier.
            :param name:    le nom de la variable à afficher en premier
                            sinon la premiere variable du premier signal.

            On la décompose de la fonction d'affichage pour avoir plus de
            flexibilité si on souhaite dériver la classe et proposer
            d'autres interfaces graphiques. Une méthode principale `plot()` doit
            d'abord créer la figure avec `make_subplot()` même si ici on utilise
            qu'un unique graphe. Cette figure est ensuite passée en agument `f`.

            Cette version crée 5 objets à l'écran :

            * `variable_dropdown`:  une liste de variables.
            * `signal_slider`:      la scrollbar correspondant aux
                                    différents signaux.
            * `previous_button`:    le bouton 'précédent'.
            * `next_button`:        le bouton 'suivant'.
            * `update_function`:    la fonction de mise à jour de 
                                    l'affichage.

            La mise à jour de l'affichage se fait par le callback 
            `update_function()`. Dans la méthode `plot()` elle est exécutée
            par l'appel à la fonction `interactive()`:

              out =  widgets.interactive(update_function,
                                        colname=variable_dropdown,
                                        sigpos=signal_slider)

            Si l'on modifie cette fonction il faudra d'abord stocker sa valeur
            puis appeler l'appeler avec les objets correspondants si on souhaite
            qu'ils restent actifs.

            :return: le dictionnaire d'éléments utiles à l'affichage décrit
                        ci-dessus.
        """

        # Une erreur s'il n'y a rien dans le fichier.
        nbmax = self.ddf.npartitions
        if nbmax==0:
            raise Affichage_Interactif_Error(self.storename, "Opset is empty.")
        
        if (pos is not None) and (pos >= 0) and (pos < nbmax) and (pos != self.sigpos):
            self.sigpos = pos
        # On relit systématiquement le fichier au début au cas où une nouvelle colonne
        # serait ajoutée.
        self.df = self.ddf.partitions[self.sigpos]

        if (name is not None):
            self.colname = get_colname(self.df.columns,name)
        if (phase is not None):
            self.phase = get_colname(self.df.columns,phase,default=None)
        
        # Définition des données à afficher.
        f.add_trace(go.Scatter(x=self.df.index, y=self.df[self.colname],name="value"),
                    row=1,col=1)
        if self.phase:
            ind = self.df[self.phase]
            f.add_trace(go.Scatter(x=self.df.index[ind], 
                                   y=self.df[self.colname][ind],
                                   name="phase",
                                   line={'color':'red'}),
                       row=1, col=1)
        
        # Description de la figure graphique.
        f.update_layout(width=500, height=400, showlegend=False)
      
        
        # ---- Callback de mise à jour de la figure ----
        def update_plot(colname, sigpos):
            """ La fonction d'interactivité avec les gadgets.
            
                Ce callback utilise la continuation de la figure `f`. 
                On fait cela pour qu'à chaque appel une nouvelle figure
                soit effectivement créée et empilée dans le notebook.
            """
            
            self.colname = colname
            name, unit = nameunit(colname)
            
            # Pour éviter de relire deux fois le signal initial.
            if sigpos != self.sigpos:
                self.sigpos = sigpos
                self.df = self.ddf.partitions[sigpos]
                

            # Mise à jour des courbes.
            f.update_traces(selector=dict(name="value"),
                            x = self.df.index, y = self.df[self.colname])
            # Affichage superposé de la phase identifiée.
            if self.phase is not None:
                ind = self.df[self.phase]
                f.update_traces(selector=dict(name="phase"),
                                x = self.df.index[ind],
                                y = self.df[self.colname][ind])

            # Mise à jour des titres et labels. 
            #print(self.records.values[sigpos])
            f.update_layout(title=self.records.values[sigpos], 
                            yaxis_title=name + '  [ ' + unit + ' ]')
        # ---- Fin du calback ----
         
            
        # Construction des gadgets interactifs.
        wd = widgets.Dropdown(options=self.df.columns,
                              value=self.colname,
                              description="Variable :")
        wbp = widgets.Button(description='Previous')
        wbn = widgets.Button(description='Next')
        ws = widgets.IntSlider(value=self.sigpos, min=0, max=nbmax-1, step=1,
                               orientation='vertical',
                               description='Record',
                               continuous_update=False,
                               layout=widgets.Layout(height='360px'))


        # ---- Callback des boutons ----
        def wb_on_click(b):
            """ Callbacks des boutons Previous et Next."""
            if b.description == 'Previous':
                if ws.value > 0:
                    ws.value -= 1
            if b.description == 'Next':
                if ws.value < ws.max:
                    ws.value += 1
        # ---- Fin du callback ----

        wbp.on_click(wb_on_click)
        wbn.on_click(wb_on_click)
        

        # Mise à jour de l'affichage.
        update_plot(self.colname, self.sigpos)
        
        # On renvoie le dictionnaire des objets graphiques.
        return dict(variable_dropdown = wd,
                    signal_slider = ws,
                    previous_button = wbp,
                    next_button = wbn,
                    update_function = update_plot)
    
    
    def plot(self,phase=None,pos=None,name=None):
        """ Affichage de l'interface.
        
            La méthode `plot()` commence par créer les différents éléments par
            un passage de ses paramètres à `make_figure()`, puis elle doit
            mettre en oeuvre l'interactivité par un appel à `interactive`
            et construire le look de l'interface en positionnant les 
            objets. Il est aussi possible de modifier le `layout`de la 
            figure.

            En entrée, les mêmes paramètres que `make_figure()`,
            et en sortie une organisation des éléments dans une boite.
            
            Il est important de créer la figure avec `make_subplots()` car
            on pourra ainsi utiliser les position des graphes dans le cas d'une
            dérivation de la classe.
        """

        f = make_subplots(rows=1, cols=1)
        f = go.FigureWidget(f)
        e = self.make_figure(f, phase,pos,name)
        out = widgets.interactive(e['update_function'], 
                                  colname=e['variable_dropdown'], 
                                  sigpos=e['signal_slider'])
        boxes = widgets.VBox([widgets.HBox([e['variable_dropdown'], 
                                            e['previous_button'], 
                                            e['next_button']]),
                              widgets.HBox([f, e['signal_slider']])])
        
        return boxes
    
    def plotc(self,phase=None,pos=None,name=None):
        """ Affichage de l'interface sans passage par FigureWidgets."""
        f = make_subplots(rows=1, cols=1)
        #f = go.FigureWidget(f)
        #self.ddf = self.create_List_Vol()
        e = self.make_figure(f, phase,pos,name)
        
        
        def update_plot_c(colname, sigpos):
            e['update_function'](colname, sigpos)
            #f.update_layout(title_text=self.Nomfichier) ## Perso
            f.show()
        
        out = widgets.interactive_output(update_plot_c, dict(
                                  colname=e['variable_dropdown'], 
                                  sigpos=e['signal_slider']))
        boxes = widgets.VBox([widgets.HBox([e['variable_dropdown'], 
                                            e['previous_button'], 
                                            e['next_button']]),
                              widgets.HBox([out, e['signal_slider']])])
        return boxes

