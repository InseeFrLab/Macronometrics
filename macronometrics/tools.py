# -*- coding: utf-8 -*-
"""
@author: QME8JI

Tools for manipulation and simulation of a model
"""
import pandas as pd
import numpy as np
import sys

from scipy.optimize import root

import importlib 
from time import time

from .tools_ts import extrapolate_series

lag_trim = pd.offsets.MonthBegin(3)

#####################################
# Manipulations formelles du modèle #
#####################################


def verif_model(model):
    """
    Vérifie le modèle :
    - nombre d'endogènes == nombre d'équations
    - déclaration des variables 
    - données présentes
    """

    # A compléter

    return


def addsym(model, status, name_sym, verbose=True):
    """
    Add a new symbol to the model

    model : A macro model
    status : "endogenous" / "exogenous" / "policy" / "coefficient"
    name_sym : name of the symbol (string)
    """
    if name_sym not in model.symboles_dict.keys():
        if status == "endogenous":
            model.name_endo_list.append(name_sym)
            model.name_endo_list.sort()

        elif status == "exogenous":
            model.name_exo_list.append(name_sym)
            model.name_exo_list.sort()

        elif status == "policy":
            model.name_policy_list.append(name_sym)
            model.name_policy_list.sort()

        elif status == "coefficient":
            model.name_coeff.append(name_sym)
            model.name_policy_list.sort()

        else:
            raise SyntaxError('Unknown status')

        model.symboles_dict[name_sym] = status

        model.is_analyzed = False
        model.is_built = False  # mise à jour du statut du modèle

        if status == "exogenous" or status == "endogenous" or status == "policy" :
            model.vall_set =  model.vall_set | {name_sym}


        if verbose:
            print(name_sym + " is defined as " + status +
                  " in the model " + model.name_mod)

    return


def delsym(model, name_sym, verbose=True):
    """
    Delete a symbol from the model
    """
    if name_sym in model.symboles_dict.keys():
        status = model.symboles_dict[name_sym]

        if status == "endogenous":
            model.name_endo_list.remove(name_sym)

        elif status == "exogenous":
            model.name_exo_list.remove(name_sym)

        elif status == "policy":
            model.name_policy_list.remove(name_sym)

        elif status == "coefficient":
            model.name_coeff.remove(name_sym)

        else:
            raise SyntaxError('Unknown status')

        del model.symboles_dict[name_sym]

        if status == "exogenous" or status == "endogenous" or status == "policy" :
            model.vall_set =  model.vall_set - {name_sym}

        model.is_analyzed = False
        model.is_built = False  # mise à jour du statut du modèle

        if verbose:
            print(name_sym + " is deleted from the model " + model.name_mod)

    return


def changesym(model, status, name_sym, verbose=True):
    """
    Change the status of a symbol
    """
    if name_sym in model.symboles_dict.keys():

        old_statut = model.symboles_dict[name_sym]

        delsym(model, name_sym, verbose=False)
        addsym(model, status, name_sym, verbose=False)

        if verbose:
            print(name_sym + " has changed from " + old_statut + " to " +
                  status + " in the model " + model.name_mod)

    else:
        raise SyntaxError('Unknown symbol')

    return


# def deleq(model, nom_eq):
#     """
#     Supprime une équation
#     """
#     if nom_eq in model.equations.keys():
#         del model.equations[nom_eq]

#         model.is_built = False  # mise à jour du statut du modèle

#     return


# def addeq(model, nom_eq, texte_eq):
#     """
#     Ajoute une équation
#     """
#     if nom_eq not in model.equations.keys():

#         model.equations[nom_eq] = texte_eq.strip()
#         model.is_built = False  # mise à jour du statut du modèle

#     else:
#         raise SyntaxError('Duplicate name')

#     return


# def changeeq(model, nom_eq, texte_eq):
#     """
#     Modifie une équation
#     """
#     if nom_eq in model.equations.keys():

#         deleq(model, nom_eq)
#         addeq(model, nom_eq, texte_eq)

#     else:
#         raise SyntaxError('Unknown name')

#     return



########################
# Simulation du modèle #
########################


def readcoeffs(filename="coefficients.csv"):
    """
    Load the values of the coefficients from a .csv file.

    Argument :
    nfilename : name of the .csv file (with the extension)

    Result :
    dictionnary name -> value
    """
    data = pd.read_csv(filename, header=None)
    # data[0] contient les noms des coefficients et data[1] les valeurs associées

    # construit un dictionnaire des données contenues dans le fichier excel
    n_lines = data.shape[0]
    dico_coeffs = dict()
    for i in range(n_lines):
        dico_coeffs[data[0][i]] = data[1][i]

    return dico_coeffs


def simulate(df_mod, val_coeff, start_date, end_date, function_name, use_jac=True, tol=1e-10, itm=1000, method='sp_root', dir="./modeles_python"):
    """
    Simulation of a model from a .py file :
    ============================================
     * the dataframe df_mod contains all the endogenous/exogenous/policy variables
     * simulation is performed between start_date and end_date (included)


    Arguments :
    ===========
     * df_mod : pandas dataframe with the time series of the model
     * val_coeff : dictionnary name : value for the coefficients of the model 
     * start_date : date of the beginning of the simulation (format YYYYQ) 
     * end_date : date of the end of the simulation (format YYYYQ)
     * function_name : name of the python function associated with the model
     * use_jac : True if the symbolic jacobian is used during the simulation

    """

    start_time = time()    # tic

    # import sys
    sys.path.insert(1, dir)

    # pour pouvoir importer un module dans l'environnement courant
    importlib.invalidate_caches()
    try:
        mdl = importlib.import_module(function_name)
    except ModuleNotFoundError:
        print('Model is not built yet. Build and write the model into a Python file')
        sys.exit()

    n_blocks = getattr(mdl, "n_blocks")  # nombre de blocs dans le modèle

    coln = getattr(mdl, "coln")  # Liste des noms de variables
    # Dictionnaire de correspondance des noms de variables
    dicovar = getattr(mdl, "dicovar")

    # coeffs = getattr(mdl, "coeffs")  # Liste des coefficients du modèle
    #    dicoeff = getattr(mdl,"dicoeff")    # dictionnaire de correspondance des coefficients

    # fonctions associées au modèle
    # fonction permettant de récupérer les endogènes de chaque bloc
    funmodel_varendo = getattr(mdl, function_name+"_varendo")
    # fonction permettant de récupérer les correspondances de chaque bloc
    funmodel_dicoendo = getattr(mdl, function_name+"_dicoendo")

    # définition des dates au format pandas
    start_date_pd = pd.to_datetime(start_date)
    end_date_pd = pd.to_datetime(end_date)

    # on copie les colones utiles dans un nouveau data frame
    # les colones sont bien ordonnées dans le data frame
    data_sim = df_mod[coln].copy()

    index_date = len(data_sim[str(data_sim.index[0]):start_date_pd])-1

    data_results = data_sim.copy()  # pour le stockage des résultats

    ix = pd.date_range(
        start=str(data_sim.index[0]), end=end_date_pd, freq="QS")
    # pour avoir l'index de dates correspondant au résultat de la simulation
    data_results = data_results.reindex(ix)

    iter_dates = pd.date_range(start=start_date_pd, end=end_date_pd, freq="QS")

    # chargement des données du modèle sous forme de tableau
    datanp = data_sim.to_numpy()  # au format np.array
    # pour stocker les résultats partiels dans un tableau
    data_result_np = np.copy(datanp)

    n_simul = len(iter_dates)

    elapsed_time = time() - start_time  # toc

    print(f"Loading the model took {elapsed_time:.3f} seconds.\n")

    start_time = time()  # tic

    for i in range(n_simul):

        for count in range(n_blocks):

            # dictionnaire nom endogène -> colonne dans le tableau
            nom_col_endo = funmodel_dicoendo(count)
            # liste des noms des endogènes
            list_var_endo = funmodel_varendo(count)

            # liste des indices des endogènes du bloc courant (A SIMPLIFIER !!!)
            list_endo = list(nom_col_endo.values())

            # Récupération des fonctions du bloc
            g = getattr(mdl, function_name+"_"+str(count))

            if use_jac:    # utilisation du jacobien symbolique
                g_jac = getattr(mdl, function_name+"_"+str(count)+"_jac")
            else:
                g_jac = False       # Jacobien approché numériquement

            x_start = np.zeros(len(list_var_endo))

            # initialisation de la méthode de résolution à la dernière date connue des endogènes

            x_start = data_result_np[index_date + i - 1, list_endo]

            # if (method == 'ggn11'):
            #     (x_res, _, _) = ggn11(x_start, g, g_jac, ftol=tol, itermax=itm,
            #                           alphamin=0.05, args=(index_date+i, data_result_np, val_coeff))
            #     for j, item in enumerate(list_var_endo):
            #         data_result_np[index_date + i, dicovar[item]
            #                        ] = x_res[j]  # mise à jour des résultats

            # else:  # on utilise fsolve à la place de root pour voir ? ...
            res_spo = root(g, x_start, args=(
                index_date+i, data_result_np, val_coeff), jac=g_jac, options={'xtol' : tol})
            # mise à jour des endogènes pour le bloc suivant à la date courante
            for j, item in enumerate(list_var_endo):
                data_result_np[index_date + i,
                                dicovar[item]] = res_spo.x[j]

    elapsed_time = time() - start_time

    print(f"The simulation of the model took {elapsed_time:.3f} secondes.\n")

    return pd.DataFrame(data=data_result_np, index=data_sim.index, columns=coln)

def simulate_cython(df_mod, val_coeff, start_date, end_date, function_name, use_jac=True, tol=1e-10, itm=1000, method='sp_root', dir="./modeles_cython"):
    """
    Simulation of a model from a cython module :
    ============================================
     * the dataframe df_mod contains all the endogenous/exogenous/policy variables
     * simulation is performed between start_date and end_date (included)


    Arguments :
    ===========
     * df_mod : pandas dataframe with the time series of the model
     * val_coeff : dictionnary name : value for the coefficients of the model 
     * start_date : date of the beginning of the simulation (format YYYYQ) 
     * end_date : date of the end of the simulation (format YYYYQ)
     * function_name : name of the python function associated with the model
     * use_jac : True if the symbolic jacobian is used during the simulation

    """

    start_time = time.time()    # tic
    
    # import sys
    sys.path.insert(1, dir)

    # pour pouvoir importer le module Cython dans l'environnement courant
    importlib.invalidate_caches()
    try:
        mdl = importlib.import_module(function_name)
    except ModuleNotFoundError:
        print('Model is not built yet. Build and compile the model into a Cython file')
        sys.exit()

    funmodel_n_blocks = getattr(mdl, function_name+"_n_blocks")
    n_blocks = funmodel_n_blocks()  # nombre de blocs dans le modèle

    funmodel_coln = getattr(mdl, function_name+"_coln")
    coln = funmodel_coln()  # Liste des noms de variables
    
    # Dictionnaire de correspondance des noms de variables
    funmodel_dicovar = getattr(mdl, function_name+"_dicovar")
    dicovar = funmodel_dicovar()

    funmodel_coeffs = getattr(mdl, function_name+"_coeffs")  # Liste des coefficients du modèle
    coeffs = funmodel_coeffs()

    # fonctions associées au modèle
    # fonction permettant de récupérer les endogènes de chaque bloc
    funmodel_varendo = getattr(mdl, function_name+"_varendo")
    # fonction permettant de récupérer les correspondances de chaque bloc
    funmodel_dicoendo = getattr(mdl, function_name+"_dicoendo")
    
    # on copie les colonnes utiles dans un nouveau data frame
    # les colonnes sont bien ordonnées dans le data frame
    data_sim = df_mod[coln].copy()
    
    # définition des dates au format pandas
    start_date_pd = pd.to_datetime(start_date)
    end_date_pd = pd.to_datetime(end_date)

    #identification du rang de la première date dans le dataframe
    index_date = len(data_sim[str(data_sim.index[0]):start_date_pd])-1

    iter_dates = pd.date_range(start=start_date_pd, end=end_date_pd, freq="QS")

    # chargement des données du modèle sous forme de tableau
    data_result_np = data_sim.to_numpy()  # au format np.array
 
    n_simul = len(iter_dates)

    elapsed_time = time.time() - start_time  # toc

    print(f"Loading the model took {elapsed_time:.3f} seconds.\n")

    start_time = time.time()  # tic

    for i in range(n_simul):

        for count in range(n_blocks):

            # dictionnaire nom endogène -> colonne dans le tableau
            nom_col_endo = funmodel_dicoendo(count)
            # liste des noms des endogènes
            list_var_endo = funmodel_varendo(count)

            # liste des indices des endogènes du bloc courant (A SIMPLIFIER !!!)
            list_endo = list(nom_col_endo.values())

            # Récupération des fonctions du bloc
            g = getattr(mdl, function_name+"_"+str(count))

            if use_jac:    # utilisation du jacobien symbolique
                g_jac = getattr(mdl, function_name+"_"+str(count)+"_jac")
            else:
                g_jac = False       # Jacobien approché numériquement

            x_start = np.zeros(len(list_var_endo))

            # initialisation de la méthode de résolution à la dernière date connue des endogènes

            x_start = data_result_np[index_date + i - 1, list_endo]


            res_spo = spo.root(g, x_start, args=(
                index_date+i, data_result_np, val_coeff), jac=g_jac, options={'xtol' : tol})
            # mise à jour des endogènes pour le bloc suivant à la date courante
            for j, item in enumerate(list_var_endo):
                data_result_np[index_date + i,
                                dicovar[item]] = res_spo.x[j]

    elapsed_time = time.time() - start_time

    print(f"The simulation of the model took {elapsed_time:.3f} secondes.\n")

    return pd.DataFrame(data=data_result_np, index=data_sim.index, columns=coln)



def simul_shock(function_name, val_coeff, data_sim, start_date_shock, end_date_shock, start_date_sim, end_date_sim, info_shock, dir="./modeles_python"):
    """
    Simulation of a shock on the dataset data_sim (containing the time series between dates start_date_shock and end_date_shock).

    The solution is computed with the function simulate. 

    info_shock is a list of dictionaries, where each item has keys :
     * variable : list of names of the shocked variables
     * type : pctge / level / chron (the type of shock)
     * value : the size of the shock (or the name of the variables in the case of a chronicle)

    The model is simulated between start_date_sim and end_date_sim.
    """

    import sys
    sys.path.insert(1, dir)

    importlib.invalidate_caches()

    try:
        importlib.import_module(function_name)
    except ModuleNotFoundError:
        print('Model is not built yet. Build and write the model into a Python file')
        sys.exit()

    data_sim_mod = data_sim.copy()

    for shock in info_shock:

        if shock["type"] == 'level':
            liste_mod = []
            for item in shock["variable"]:
                liste_mod.append(
                    [item, 'affine', [start_date_shock, end_date_shock], 1, shock["value"]])
                data_sim_mod = extrapolate_series(
                    data_init=data_sim_mod, liste_series=liste_mod)

        elif shock["type"] == 'pctge':
            liste_mod = []
            for item in shock["variable"]:
                liste_mod.append(
                    [item, 'affine', [start_date_shock, end_date_shock], 1+0.01*shock["value"], 0])
                data_sim_mod = extrapolate_series(
                    data_init=data_sim_mod, liste_series=liste_mod)

        else:
            for item in shock["variable"]:
                data_sim_mod[item] = shock["value"][item]

    return simulate(data_sim_mod, val_coeff, start_date_sim, end_date_sim, function_name)
