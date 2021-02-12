# -*- coding: utf-8 -*-
"""
@author: QME8JI
"""

from re import sub  # Pour les expressions régulières
from copy import deepcopy 

from .equation import Equation  # gestion du parseur
from .symbolic import Block   # construction des fonctions de résolution
from .analyze import analyze_model, write_yaml_file

from time import time

from .graph import Edge
from .getDst import getDst, getSrc
from .DulmageMendelsohnDecomposition import DulmageMendelsohnDecomposition


def unique_list(liste):
    """
    Return a list without duplicate entries.
    """
    return list(set(liste))


class Model():
    """
    Definition of the main class to manage a Model object.

    The Model class :
    - performs the lexing stage (model.lexer),
    - makes preliminary computations (model.prelim), 
    - contains a deepcopy tool (model.copy).

    """

    def __repr__(self):
        return "Macroeconometric model : {}".format(self.name_mod)

    def __init__(self):
        """
        Class builder
        """
    # nom du modèle
        self.name_mod = ""

        # listes globales des noms des variables, des coefficients et des étiquettes d'équations
        self.name_endo_list = []
        self.name_exo_list = []
        self.name_policy_list = []
        self.name_coeff_list = []
        self.name_eq_list = []

        # pour avoir l'ensemble des variables et créer la fonction du modèle
        self.coln_list = []

        # équations et coefficients

        self.eq_text_dict = dict()  # correspondance nom_eq : texte
        self.eq_obj_dict = dict()   # pour utiliser l'objet equation identifiée par son nom
        self.eq_obj_dict_number = dict()
        self.coeffs_dict = dict()

        self.dicovar = dict()       # dictionnaires globaux (variables, coefficients)
        self.dicoeff = dict()

        self.var_eq_dict = dict()   # correspondance variable : équation

        # correspondances numéro d'équation : variables (exo / policy / endo / coeff) + endogènes retardées (pour l'analyse du modèle)
        self.eq_exo_dict = dict()
        self.eq_policy_dict = dict()
        self.eq_endo_dict = dict()
        self.eq_coeff_dict = dict()
        self.eq_endo_lag_dict = dict()

        self.symboles_dict = dict()         # statut des symboles (pour les variantes)
        self.vall_set = set()               # ensemble des variables

        # texte des fonctions
        self.fun_text = ""
        self.jac_text = ""

        self.n_eq = 0   # nomnbre d'équations
        # le modèle est construit lorsqu'on a écrit les fonctions de résolution
        self.is_built = False
        # le modèle est analysé lorsqu'on a construit l'ensemble des objets nécessaires
        self.is_analyzed = False

    def lexer(self, model_code_filename):
        """
        Lexer of model code file.

        Argument :
        model_code_filename : name of the file  (Troll format)

        Warning : the file is supposed to be valid.
        """

        self.nom_fichier_code = model_code_filename
        str_file = ""

        # Lecture du fichier (en utf 8)
        with open(self.nom_fichier_code, "r", encoding="utf-8") as f:
            for line in f.readlines():
                str_file += line.split("//")[0]

        # prétraitement
        str_file = str_file.strip()
        str_file = str_file.lower()
        str_file = str_file.replace("\t", " ")
        str_file = str_file.replace("\n", " ")

        # suppression des commentaires et simplification des espaces
        # expression régulière magique !
        str_file2 = sub(
            r"/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/", " ", str_file)
        str_file2 = sub(" +", " ", str_file2)

        self.list_file = str_file2.strip().split(";")

        compte_eq = 0  # compte le nombre d'équations dans le fichier du modèle
        for content in self.list_file:

            indic = content.strip().split()

            if len(indic) > 0:

                if indic[0] == 'filemod':
                    # recupère le nom du modele en fin de fichier
                    self.name_mod = indic[1]

                elif indic[0] == 'addsym':
                    # traitement des variables
                    variables = content.split(",")

                    for item in variables:
                        var_list_raw = item.split()
                        if var_list_raw[0] == 'addsym':
                            self.affecte_var(var_list_raw[1:])
                        else:
                            self.affecte_var(var_list_raw)

                elif indic[0] == "addeq":
                    # traitement des équations
                    eqns = content.split(",")[1:]
                    for eq in eqns:
                        if len(eq) > 0:
                            eq_split = eq.strip().split(":")
                            lhs = eq_split[0].strip()
                            rhs = ":".join(eq_split[1:])
                            if lhs not in self.eq_text_dict.keys():
                                self.eq_text_dict[lhs] = [
                                    compte_eq, rhs.strip()]
                                self.eq_obj_dict[lhs] = Equation(
                                    name_eq=lhs, text_eq=rhs.strip(), num_eq=compte_eq)
                                self.eq_exo_dict[compte_eq] = set()
                                self.eq_policy_dict[compte_eq] = set()
                                self.eq_endo_dict[compte_eq] = set()
                                self.eq_endo_lag_dict[compte_eq] = set()
                                self.eq_coeff_dict[compte_eq] = set()
                                compte_eq += 1

                            else:  # cas de doublons dans les noms d'équations
                                self.eq_text_dict[lhs +
                                                  "_bis"] = [compte_eq, rhs.strip()]
                                self.eq_obj_dict[lhs+"_bis"] = Equation(
                                    name_eq=lhs+"_bis", text_eq=rhs.strip(), num_eq=compte_eq)
                                self.eq_exo_dict[compte_eq] = set()
                                self.eq_policy_dict[compte_eq] = set()
                                self.eq_endo_dict[compte_eq] = set()
                                self.eq_endo_lag_dict[compte_eq] = set()
                                self.eq_coeff_dict[compte_eq] = set()
                                compte_eq += 1

        # on maintient les listes globales des noms des variables
        self.name_eq_list = list(self.eq_text_dict.keys())
        self.n_eq = compte_eq

        self.name_endo_list = unique_list(self.name_endo_list)
        self.name_exo_list = unique_list(self.name_exo_list)
        self.name_coeff_list = unique_list(self.name_coeff_list)
        self.name_policy_list = unique_list(self.name_policy_list)

        # ensemble des variables
        self.vall_set = set(self.name_endo_list) | set(
            self.name_exo_list) | set(self.name_policy_list)

        return

    def affecte_var(self, tab):
        """
        Allocation of the names of the variables from the header of the file.

        Argument : 
        tab : an array of strings from the lexer.

        Warning : some exogenous variables may be definend implicitely or directly in the code.
        """
        if tab[0] == "policy":
            self.name_policy_list = tab[1:]
        elif tab[0] == "exogenous":
            self.name_exo_list = tab[1:]
        elif tab[0] == "endogenous":
            self.name_endo_list = tab[1:]
        elif tab[0] == "coefficients":
            self.name_coeff_list = tab[1:]
        return

    def sort_name_var(self):
        """
        Sort the lists of the variables and coefficients names.

        """

        self.name_endo_list.sort()
        self.name_exo_list.sort()
        self.name_coeff_list.sort()
        self.name_policy_list.sort()

        return

    def prelim(self):
        """
        Preliminary computations and initialization of data structures for solving the model

        Argument : None

        """
        # self.date = date

        # contient les lignes correspondant aux equations nettoyées du fichier source
        self.lines = []
        for v in list(self.eq_text_dict.keys()):
            self.lines.append(v+':'+self.eq_text_dict[v][1])

        # contient les équations sans leur étiquette
        self.lines_eq = []
        for v in self.eq_text_dict.values():
            self.lines_eq.append(v[1])

        self.sort_name_var()

        print("The model has ", self.n_eq, " equations.\n")
        print(len(self.name_endo_list), " endogenous variables declared.\n")

        return

    ###################################################
    # Début des fonctions permettant la décomposition #
    #        de Dulmage - Mendelsohn  (BoJ)           #
    ###################################################

    def setup(self):
        """
        Setup of some useful data structures for the Dulmage-Mendelsohn decomposition

        Argument : None

        """

        # Pour avoir la clé de passage entre le nom d'une variable endogène et un indice chiffré

        # correspondance nom : indice / identique à dicovar
        self.d = {v: i for i, v in enumerate(sorted(self.name_endo_list))}
        # correspondance indice : nom
        self.dr = {i: v for i, v in enumerate(sorted(self.name_endo_list))}

        return

    def analyze_structure(self):
        """
        Analysis of the block structure of a model.

        Argument : None
        """
        self.setup()
        self.construction_graph()
        self.dm_decomp()
        self.construct_gb()            # optional
        self.construct_blocked_model()  # optional

    def construction_graph(self):
        """
        Computation of a graph of dependence between exogenous variables

        """

        # le graphe est défini par sa matrice d'adjacence
        self.g = [[] for n in range(len(self.d))]

        for e in range(len(self.endoss)):
            for v in sorted(self.endoss[e]):
                self.g[e].append(Edge(e, self.d[v]))

        return

    def dm_decomp(self):
        """
        Finest blocks decomposition
        """

        # detect finest block structure (core routine)
        self.rss = []
        self.css = []
        DulmageMendelsohnDecomposition(self.g, self.rss, self.css)

        return

    def find_css_block(self, v):
        # find css_block in which variable v joins
        for k in range(len(self.rss)):
            if v in self.css[k]:
                return k

    def construct_gb(self):
        # construct graph of dm-decomped blocks' dependency
        nb = len(self.rss)
        self.endossb = [set() for n in range(nb)]
        for k in range(nb):
            for e in self.rss[k]:
                self.endossb[k] |= {
                    self.find_css_block(v.dst) for v in self.g[e]}

        self.gb = [[] for n in range(nb)]
        for k in range(nb):
            for v in self.endossb[k]:
                self.gb[k].append(Edge(k, v))

    def construct_blocked_model(self):
        # each block's endos set, exogs set, equation list, determined vars set
        self.endobss = [{v for r in rs for v in self.endoss[r]}
                        for rs in self.rss]
        self.exogbss = [{v for r in rs for v in self.exogss[r]}
                        for rs in self.rss]
        self.linebss = [[self.lines[r] for r in rs] for rs in self.rss]
        self.lineqbss = [[self.lines_eq[r] for r in rs] for rs in self.rss]
        self.determined = [{self.dr[c] for c in cs} for cs in self.css]

    def classify_vars(self, var):
        # pre = v joins predetermined blocks before var
        # sim = v joins simultaneously determined block with var
        # pos = v joins postdetermined blocks after var ('burasagari' in Japanese)
        # iso = v joins isolated block from var block
        v = self.d[var]
        simb = self.find_css_block(v)
        preb = getDst(self.gb, [simb])
        posb = getSrc(self.gb, [simb])
        preb.remove(simb)
        posb.remove(simb)
        pre = {self.dr[v] for b in preb for v in self.css[b]}
        pos = {self.dr[v] for b in posb for v in self.css[b]}
        sim = {self.dr[v] for v in self.css[simb]}
        iso = set(self.name_endo_list) - pre - pos - sim
        return [pre, sim, pos, iso]

    def build_modeltext_blocked(self):
        """
        Compute the structure of blocks and the associated text of equations sets.

        """
        self.endobss_left_eq = [{l.split(':')[0]
                                 for l in ls} for ls in self.linebss]
        self.vendo_left_eq = {v for vs in self.endobss_left_eq for v in vs}
        self.vexog_left_eq = self.vall_set - self.vendo_left_eq

        # calc exog2endo and endo2exog in each finest block
        nb = len(self.rss)
        self.solved = set()
        self.determined = [set() for n in range(nb)]
        self.exog2endo = [set() for n in range(nb)]
        self.endo2exog = [set() for n in range(nb)]
        for b in range(nb):
            self.determined[-b-1] = self.endobss[-b-1] - self.solved
            #print('determined : '+str(self.determined[-b-1]))
            # exog for EViews, mathematically endo determined in the block
            self.exog2endo[-b-1] = self.determined[-b-1] - \
                self.endobss_left_eq[-b-1]
            #print('exog2endo : '+str(self.exog2endo[-b-1]))
            # endo for EViews, mathematically not endo determined in the block
            self.endo2exog[-b-1] = self.endobss_left_eq[-b-1] - \
                self.determined[-b-1]
            #print('endo2exog : '+str(self.endo2exog[-b-1]))
            self.solved |= self.endobss[-b-1]
            #print('solved : '+str(self.solved))

        # merge until endo2exog and exog2endo are empty
        self.exog2endo_mg = [[] for n in range(nb)]
        self.endo2exog_mg = [[] for n in range(nb)]
        self.rs_mg = [[] for n in range(nb)]
        n_mg, prev = 0, False
        for b in range(nb):
            # split at non empty exog2endo and V0 and Vinf
            if self.exog2endo[-b-1] or prev or b == 1 or b == nb-1:
                n_mg += 1
            self.exog2endo_mg[n_mg] = self.exog2endo[-b-1]
            self.endo2exog_mg[n_mg] = self.endo2exog[-b-1]
            self.rs_mg[n_mg] += self.rss[-b-1][::-1]  # reverse
            prev = True if self.exog2endo[-b-1] else False
        self.exog2endo_mg = self.exog2endo_mg[0:n_mg+1]
        self.endo2exog_mg = self.endo2exog_mg[0:n_mg+1]
        self.rs_mg = self.rs_mg[0:n_mg+1]

    # merged blocked model for writing modeltext file
        self.fs = dict()
        print("The block decomposition has " +
              str(len(self.rs_mg)-2) + ' blocks.\n')
        self.fs[0] = [0, [], set(), set()]
        exo = set()
        for n in range(1, len(self.rs_mg)-1):
            self.fs[n] = [len(self.rs_mg[n]), [], set(), set()]
            for r in range(len(self.rs_mg[n])):
                exo = exo.union(self.fs[n-1][2])
                self.fs[n][1].append(self.lines_eq[self.rs_mg[n][r]])
                self.fs[n][2] = self.fs[n][2].union(
                    self.endoss[self.rs_mg[n][r]])-exo
                self.fs[n][3] = self.fs[n][3].union(
                    exo).union(self.exogss[self.rs_mg[n][r]])
                exo = exo.union(self.fs[n][3])

        # Ecriture de la décomposition par blocs à l'aide des indices des équations (et non du texte)

        self.fsix = dict()
        self.fsix[0] = [0, [], set(), set()]
        exo = set()
        for n in range(1, len(self.rs_mg)-1):
            # premier élément : taille du bloc
            self.fsix[n] = [len(self.rs_mg[n]), [], set(), set()]
            for r in range(len(self.rs_mg[n])):
                exo = exo.union(self.fsix[n-1][2])
                # numéro des équations du bloc
                self.fsix[n][1].append(self.rs_mg[n][r])
                self.fsix[n][2] = self.fsix[n][2].union(
                    self.endoss[self.rs_mg[n][r]])-exo  # endogènes du bloc
                self.fsix[n][3] = self.fsix[n][3].union(exo).union(
                    self.exogss[self.rs_mg[n][r]])  # exogènes du bloc
                exo = exo.union(self.fsix[n][3])

    ################################################
    # Fin des fonctions de décomposition par blocs #
    ################################################

    def build_model(self, function_name, dir="./modeles_python", prod="python"):
        """
        Fill some useful data structures and performs the Dulmage - Mendelsohn block decomposition.
         * performs an analysis of the model 
         * generates Python code

        Argument : aucun
        """

        start_time = time()    # tic

        analyze_model(self)  # Les équations du modèle sont analysées ...

        self.eq_obj_dict_number = {
            self.eq_obj_dict[k].num_eq: self.eq_obj_dict[k] for k in self.eq_obj_dict.keys()}

        # Construction des dictionnaires liant variables et équations qui les contiennent
        # pour utiliser la décomposition D-M

        # dictionnaire nom de variable -> ens. d'equations
        self.varendo_eq_dict = {
            k: v for k, v in self.var_eq_dict.items() if k in self.name_endo_list}
        self.varexo_eq_dict = {k: v for k, v in self.var_eq_dict.items() if k in (
            self.name_exo_list+self.name_policy_list)}

        endoss = [set()]*self.n_eq
        for k, v in self.varendo_eq_dict.items():
            for i in v:
                endoss[i] = endoss[i] | {k}

        self.endoss = endoss

        exogss = [set()]*self.n_eq
        for k, v in self.varexo_eq_dict.items():
            for i in v:
                exogss[i] = exogss[i] | {k}

        self.exogss = exogss

        self.analyze_structure()  # Analyse la structure en blocs du modèle

        self.build_modeltext_blocked()  # calcul des blocs

        elapsed_time = time() - start_time  # toc

        print(f"The block decomposition took {elapsed_time:.3f} seconds.\n")

        # pour stocker le texte de chaque bloc en vue de produire un fichier python
        start_time = time()    # tic

        liste_string_func_block = []

        for i in range(len(self.fsix)):

            if self.fsix[i][0] != 0:

                # liste des équations du bloc
                list_eq_block = self.fsix[i][1]
                # liste des endogènes du bloc
                list_endo_block = list(self.fsix[i][2])
                # ensemble des exogènes du bloc
                list_exo_block = list(self.fsix[i][3])

                block = Block(self, list_eq_block,
                              list_endo_block, list_exo_block, i)

                block.translate_block_python()

                liste_string_func_block.append(
                    [block.fun_text, list(self.fsix[i][2]), block.jac_text])
                # on ajoute le jacobien en dernière position pour ne pas modifier les programmes existants / modif BF

        self.model_fun_block = liste_string_func_block

        elapsed_time = time() - start_time  # toc

        print(f"Building  the function took {elapsed_time:.3f} seconds.\n")

        # Correspondance symbole -> statut (pour le calage et le calcul des variantes)

        for item in self.name_endo_list:
            self.symboles_dict[item] = "endogenous"

        for item in self.name_exo_list:
            self.symboles_dict[item] = "exogenous"

        for item in self.name_policy_list:
            self.symboles_dict[item] = "policy"

        for item in self.name_coeff_list:
            self.symboles_dict[item] = "coefficient"

        self.is_built = True  # Le modèle est désormais construit.

        # if prod == "python":
        #     self.write_model(function_name, dir)
        #     write_yaml_file(self, function_name+".yaml", dir)
        # else:
        #     self.write_model_cython(function_name, dir='./modeles_cython')
        #     write_yaml_file(self, function_name+".yaml", dir='./modeles_cython')

        self.write_model(function_name, dir)
        write_yaml_file(self, function_name+".yaml", dir)

        return

    def copy(self):
        """
        Return a copy (without reference / deepcopy) of the model.
        """

        return deepcopy(self)

    def write_model(self, function_name, dir="./modeles_python"):
        """
        Creation of a Python file containing the text of the model functions.

        Arguments :
        ===========
        model : a preliminary analyzed model
        function_name : name of the function (string)
        dir : name of the directory (string)

        Result :
        ========
         * A .py file with the functions (for each block) and their jacobians.

        """

        if (not self.is_built):

            raise ValueError("The model is not built.")

        n_blocks = len(self.model_fun_block)

        # Structures de données pour le traitement d'un bloc

        list_block_fun = []
        list_block_varendo = []
        list_block_jac = []
        list_block_dicoendo = []

        # sous la forme [ texte , var_endo , texte_jac ]
        for item in self.model_fun_block:
            list_block_fun.append(item[0])
            list_block_varendo.append(item[1])
            list_block_jac.append(item[2])

            # pour chaque bloc, on construit un dictionnaire de correspondance
            # endogène -> numero
            nom_col_endo = {}
            # on considère les endogènes du bloc courant
            for endocourant in item[1]:
                # correspondance nom / indice
                nom_col_endo[endocourant] = self.dicovar[endocourant]

            list_block_dicoendo.append(nom_col_endo)

        # texte de la fonction déterminant les endogènes pour chaque bloc

        text_varendo = 'def ' + function_name.strip() + '_varendo(num_block): \n'
        text_varendo += '\t"""\n \tFonction produite automatiquement pour la résolution du modèle \n\n'
        text_varendo += '\tDétermine les endogènes associées à chaque bloc \n'
        text_varendo += '\t\n\tArguments : \n'
        text_varendo += '\t\tnum_block : numéro du bloc (décomposition de Dulmage-Mendelsohn) \n'
        text_varendo += '\t\n\t""" \n'
        text_varendo += '\tlist_block_varendo = ['

        for item in list_block_varendo:
            text_varendo += str(item) + " , \\\n\t\t"

        text_varendo += "] \n"

        text_varendo += '\treturn list_block_varendo[num_block] \n'

        # texte de la fonction donnant la correspondance bloc -> endogènes

        text_dicoendo = 'def ' + function_name.strip() + '_dicoendo(num_block): \n'
        text_dicoendo += '\t"""\n \tFonction produite automatiquement pour la résolution du modèle \n\n'
        text_dicoendo += '\tDétermine les correspondances des endogènes associées à chaque bloc \n'
        text_dicoendo += '\t\n\tArguments : \n'
        text_dicoendo += '\t\tnum_block : numéro du bloc (décomposition de Dulmage-Mendelsohn) \n'
        text_dicoendo += '\t\n\t""" \n'
        text_dicoendo += '\tlist_block_dicoendo = ['

        for item in list_block_dicoendo:
            text_dicoendo += str(item) + " , \\\n\t\t"

        text_dicoendo += "] \n"

        text_dicoendo += '\treturn list_block_dicoendo[num_block] \n'

        # préambule de la fonction associée à un bloc

        text_fun_pre = 'def ' + function_name.strip()
        text_fun = '(x,t,data,coeff): \n'
        text_fun += '\t"""\n\tFonction produite automatiquement pour la résolution du modèle \n\n'
        text_fun += '\tBloc représenté par la fonction F telle que F(x)=0 \n'
        text_fun += '\t\n\tArguments : \n'
        text_fun += '\t\tx : vecteur de variables endogènes contemporaines \n'
        text_fun += '\t\tt : date courante (dans le tableau de données) \n'
        text_fun += '\t\tdata : tableau numpy contenant les données du modèle \n'
        text_fun += '\t\n\t""" \n'

        text_jac_pre = 'def ' + function_name.strip()

        text_jac = '_jac(x,t,data,coeff): \n'
        text_jac += '\t"""\n\tFonction produite automatiquement pour la résolution du modèle \n\n'
        text_jac += '\tJacobienne de la fonction associée au bloc \n'
        text_jac += '\t\n\tArguments : \n'
        text_jac += '\t\tx : vecteur de variables endogènes contemporaines \n'
        text_jac += '\t\tt : date courante (dans le tableau de données) \n'
        text_jac += '\t\tdata : tableau numpy contenant les données du modèle \n'
        text_jac += '\t\n\t""" \n'

        # Ecriture du fichier

        with open(dir+"/" + function_name+".py", "w+", encoding='utf-8') as out_file:

            out_file.write("import numpy as np\n")
            out_file.write("from math import log, exp\n\n")

            out_file.write("import numpy as np\n\n")

            out_file.write("# Nombre de blocs du modèle\n")
            out_file.write("n_blocks = " + str(n_blocks) + " \n\n")
            out_file.write("# Liste des noms de variables\n")
            out_file.write("coln = " + str(self.coln_list) + " \n\n")
            out_file.write(
                "# Dictionnaire de correspondance des noms de variables\n")
            out_file.write("dicovar = " + str(self.dicovar) + " \n\n")
            out_file.write("# Liste des noms de coefficients\n")
            out_file.write("coeffs = " + str(self.name_coeff_list) + " \n\n")
            # out_file.write("# Dictionnaire de correspondance des noms de coefficients\n")
            # out_file.write("dicoeff = " + str(self.dicoeff)+ " \n\n")

            out_file.write(text_varendo)
            out_file.write("\n")

            out_file.write(text_dicoendo)
            out_file.write("\n")

            for i, item in enumerate(list_block_fun):
                out_file.write(text_fun_pre+"_"+str(i)+text_fun+item)
                out_file.write('\treturn f')
                out_file.write("\n\n")

                out_file.write(text_jac_pre+"_"+str(i) +
                               text_jac+list_block_jac[i])
                out_file.write('\treturn df')
                out_file.write("\n\n")

        return

    # def write_model_cython(self, function_name, dir="./modeles_cython"):
    #     """
    #     Creation of a Cython file containing the text of the model functions.

    #     Arguments :
    #     ===========
    #     model : a preliminary analyzed model
    #     function_name : name of the function (string)
    #     dir : name of the directory (string)

    #     Result :
    #     ========
    #      * A .pyx file with the functions (for each block) and their jacobians.

    #     """

    #     if (not self.is_built):

    #         raise ValueError("The model is not built.")

    #     n_blocks = len(self.model_fun_block)

    #     # Structures de données pour le traitement d'un bloc

    #     list_block_fun = []
    #     list_block_varendo = []
    #     list_block_jac = []
    #     list_block_dicoendo = []

    #     # sous la forme [ texte , var_endo , texte_jac ]
    #     for item in self.model_fun_block:
    #         list_block_fun.append(item[0])
    #         list_block_varendo.append(item[1])
    #         list_block_jac.append(item[2])

    #         # pour chaque bloc, on construit un dictionnaire de correspondance
    #         # endogène -> numero
    #         nom_col_endo = {}
    #         # on considère les endogènes du bloc courant
    #         for endocourant in item[1]:
    #             # correspondance nom / indice
    #             nom_col_endo[endocourant] = self.dicovar[endocourant]

    #         list_block_dicoendo.append(nom_col_endo)

    #     # texte de la fonction déterminant les endogènes pour chaque bloc

    #     text_varendo = 'cpdef list ' + function_name.strip() + '_varendo(int num_block): \n'
    #     text_pyd = 'cpdef list ' + function_name.strip() + '_varendo(int num_block)\n\n'
    #     text_varendo += '\t"""\n \tFonction produite automatiquement pour la résolution du modèle \n\n'
    #     text_varendo += '\tDétermine les endogènes associées à chaque bloc \n'
    #     text_varendo += '\t\n\tArguments : \n'
    #     text_varendo += '\t\tnum_block : numéro du bloc (décomposition de Dulmage-Mendelsohn) \n'
    #     text_varendo += '\t\n\t""" \n'
    #     text_varendo += '\tcdef list list_block_varendo = ['

    #     for item in list_block_varendo:
    #         text_varendo += str(item) + " , \\\n\t\t"

    #     text_varendo += "] \n"

    #     text_varendo += '\treturn list_block_varendo[num_block] \n'

    #     # texte de la fonction donnant la correspondance bloc -> endogènes

    #     text_dicoendo = 'cpdef dict ' + \
    #         function_name.strip() + '_dicoendo(int num_block): \n'
    #     text_pyd += 'cpdef dict ' + \
    #         function_name.strip() + '_dicoendo(int num_block)\n\n'
    #     text_dicoendo += '\t"""\n \tFonction produite automatiquement pour la résolution du modèle \n\n'
    #     text_dicoendo += '\tDétermine les correspondances des endogènes associées à chaque bloc \n'
    #     text_dicoendo += '\t\n\tArguments : \n'
    #     text_dicoendo += '\t\tnum_block : numéro du bloc (décomposition de Dulmage-Mendelsohn) \n'
    #     text_dicoendo += '\t\n\t""" \n'
    #     text_dicoendo += '\tcdef list list_block_dicoendo = ['

    #     for item in list_block_dicoendo:
    #         text_dicoendo += str(item) + " , \\\n\t\t"

    #     text_dicoendo += "] \n"

    #     text_dicoendo += '\treturn list_block_dicoendo[num_block] \n'

    #     # préambule de la fonction associée à un bloc

    #     text_fun_pre = 'cpdef np.ndarray[DTYPE_t, ndim=1] ' + \
    #         function_name.strip()
    #     text_fun = '(np.ndarray[DTYPE_t, ndim=1] x, int t, np.ndarray[DTYPE_t, ndim=2] data, dict coeff): \n'
    #     text_fun += '\t"""\n\tFonction produite automatiquement pour la résolution du modèle \n\n'
    #     text_fun += '\tBloc représenté par la fonction F telle que F(x)=0 \n'
    #     text_fun += '\t\n\tArguments : \n'
    #     text_fun += '\t\tx : vecteur de variables endogènes contemporaines \n'
    #     text_fun += '\t\tt : date courante (dans le tableau de données) \n'
    #     text_fun += '\t\tdata : tableau numpy contenant les données du modèle \n'
    #     text_fun += '\t\n\t""" \n'

    #     text_jac_pre = 'cpdef np.ndarray[DTYPE_t, ndim=2] ' + \
    #         function_name.strip()
    #     text_jac = '_jac(np.ndarray[DTYPE_t, ndim=1] x, int t, np.ndarray[DTYPE_t, ndim=2] data, dict coeff): \n'
    #     text_jac += '\t"""\n\tFonction produite automatiquement pour la résolution du modèle \n\n'
    #     text_jac += '\tJacobienne de la fonction associée au bloc \n'
    #     text_jac += '\t\n\tArguments : \n'
    #     text_jac += '\t\tx : vecteur de variables endogènes contemporaines \n'
    #     text_jac += '\t\tt : date courante (dans le tableau de données) \n'
    #     text_jac += '\t\tdata : tableau numpy contenant les données du modèle \n'
    #     text_jac += '\t\n\t""" \n'

    #     # Ecriture du fichier

    #     with open(dir+"/" + function_name+".pyx", "w+", encoding='utf-8') as out_file:

    #         out_file.write("from libc.math cimport log, exp\n\n")
    #         out_file.write("import numpy as np\n")
    #         out_file.write("cimport numpy as np\n\n")

    #         out_file.write("# Nombre de blocs du modèle\n")
    #         out_file.write(
    #             "cpdef int " + function_name.strip() + "_n_blocks(): \n")
    #         out_file.write("\treturn " + str(n_blocks) + " \n\n")

    #         out_file.write("# Liste des noms de variables\n")
    #         out_file.write("cpdef list " +
    #                        function_name.strip() + "_coln(): \n")
    #         out_file.write("\tcdef list coln = " +
    #                        str(self.coln_list) + " \n\n")
    #         out_file.write("\treturn coln \n\n")

    #         out_file.write(
    #             "# Dictionnaire de correspondance des noms de variables\n")
    #         out_file.write("cpdef dict " +
    #                        function_name.strip() + "_dicovar():\n")
    #         out_file.write("\tcdef dict dicovar = " +
    #                        str(self.dicovar) + " \n\n")
    #         out_file.write("\treturn dicovar \n\n")

    #         out_file.write("# Liste des noms de coefficients\n")
    #         out_file.write("cpdef list " +
    #                        function_name.strip() + "_coeffs():\n")
    #         out_file.write("\tcdef list coeffs = " +
    #                        str(self.name_coeff_list) + " \n")
    #         out_file.write("\treturn coeffs \n\n")

    #         out_file.write(text_varendo)
    #         out_file.write("\n")

    #         out_file.write(text_dicoendo)
    #         out_file.write("\n")

    #         for i, item in enumerate(list_block_fun):
    #             out_file.write(text_fun_pre+"_"+str(i)+text_fun + item + "\n\n")
    #             out_file.write("\tcdef np.ndarray[DTYPE_t, ndim=1] res = f\n\n") 
    #             out_file.write("\treturn res")
    #             out_file.write("\n\n")

    #             out_file.write(text_jac_pre+"_"+str(i) +
    #                            text_jac+list_block_jac[i] + "\n\n")
    #             out_file.write("\tcdef np.ndarray[DTYPE_t, ndim=2] res = df\n\n") 
    #             out_file.write("\treturn res")
    #             out_file.write("\n\n")

    #     with open(dir+"/" + function_name+".pxd", "w+", encoding='utf-8') as out_file_2:

    #         out_file_2.write("import numpy as np\ncimport numpy as np\n\n")
            
    #         out_file_2.write("ctypedef np.double_t DTYPE_t\n\n")

    #         out_file_2.write(
    #                 "cpdef int " + function_name.strip() + "_n_blocks() \n\n")

    #         out_file_2.write("cpdef list " +
    #                          function_name.strip() + "_coln() \n\n")

    #         out_file_2.write("cpdef dict " +
    #                          function_name.strip() + "_dicovar()\n")

    #         out_file_2.write("cpdef list " +
    #                          function_name.strip() + "_coeffs()\n\n")

    #         out_file_2.write(text_pyd)

    #         for i, item in enumerate(list_block_fun):
    #                 out_file_2.write(
    #                     text_fun_pre+"_"+str(i)+'(np.ndarray[DTYPE_t, ndim=1] x, int t, np.ndarray[DTYPE_t, ndim=2] data, dict coeff)\n\n')
    #                 out_file_2.write(text_jac_pre+"_"+str(i) +
    #                              '_jac(np.ndarray[DTYPE_t, ndim=1] x, int t, np.ndarray[DTYPE_t, ndim=2] data, dict coeff)\n\n')

    #     with open(dir+"/setup_" + function_name+".py", "w+", encoding='utf-8') as out_file_3:

    #         out_file_3.write("from setuptools import setup\n\n" +
    #                          "from Cython.Build import cythonize\n\n" +
    #                          "from sys import version_info\n\n" +
    #                          "import numpy\n\n" +
    #                          "setup(\n" +
    #                          "ext_modules=cythonize('"+function_name+".pyx',force=True, compiler_directives={'language_level' : version_info[0]}),\n" +
    #                          'include_dirs=[numpy.get_include()],\n' +
    #                          ')'
    #                          )

    #     # import os

    #     # os.chdir(dir)

    #     # os.system("$ python setup_"+function_name+".py build_ext --inplace")

    #     return

