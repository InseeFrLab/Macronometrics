# -*- coding: utf-8 -*-
"""
@author: QME8JI
"""


class Equation():
    """
    Definition of an Equation object

    """

    def __repr__(self):
        texte = "Equation : {} \n".format(self.name_eq)
        texte += "Texte : {} \n".format(self.text_eq)
        texte += "Numéro : {} \n".format(self.num_eq)
        texte += "Coefficients : {} \n".format(self.coeff_name_list)
        texte += "Exogènes : {} \n".format(self.exo_name_list)
        texte += "Policy : {} \n".format(self.policy_name_list)
        texte += "Endogènes : {} \n".format(self.endo_name_list)
        return texte

    def __init__(self, name_eq, text_eq, num_eq):
        """
        name_eq : name of the equation (in the model text)
        text_eq : text of the equation
        num_eq = equation index
        """

        self.name_eq = name_eq  # nom de l'équation
        self.text_eq = text_eq  # texte de l'équation
        self.num_eq = num_eq    # numéro de l'équation

        # self.num_block = None   # numéro du bloc contenant l'équation

        # dictionnaire coefficients de l'équation -> indice global
        self.coeff_eq_dict = dict()
        self.coeff_name_list = []    # nom des coefficients de l'équation

        # correspondance variables / données

        self.exo_eq_dict = dict()    # dictionnaire exogènes de l'équation -> indice global
        self.exo_name_list = []      # nom des exogènes de l'équation

        # dictionnaire endogènes contemporaines de l'équation -> indice global
        self.endo_eq_dict = dict()
        self.endo_name_list = []     # nom des endogènes de l'équation

        self.policy_eq_dict = dict()  # dictionnaire policy -> indice
        self.policy_name_list = []

        # arbres syntaxiques de l'équation et de ses dérivées
        self.tree_eq = None     # arbre syntaxique de l'équation
        self.tree_diff = None   # arbre syntaxique dérivé
        # self.tree_diff_dict = dict() # dictionnaire nom des endogènes contemporaines -> arbre dérivé

    def print_tree(self, deriv=False):
        """
        Print the trees
        """
        if deriv:
            print(self.tree_diff.pretty())
        else:
            print(self.tree_eq.pretty())
        return
