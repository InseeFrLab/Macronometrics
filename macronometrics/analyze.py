# -*- coding: utf-8 -*-
"""
@author: QME8JI
"""

from .trollparser import parser  # Parseur de syntaxe Troll
from lark import Tree, Token
from time import time
from yaml import dump    # pour l'écriture du fichier yaml


def unique_list(liste):
    """
    Return a list of unique elements
    """
    return sorted(list(set(liste)))


def analyze_model(model):
    """
    Analyze the model before  parsing :
        * Compute the syntaxic trees for each equation
        * Update the sets of variables and the dictionaries

    """

    model.prelim()  # pour avoir les variables ordonnées

    # unpack
    name_coeff_list = model.name_coeff_list
    name_endo_list = model.name_endo_list
    name_exo_list = model.name_exo_list
    name_policy_list = model.name_policy_list

    # coln_list = model.coln_list

    var_eq_dict = model.var_eq_dict

    eq_exo_dict = model.eq_exo_dict
    eq_policy_dict = model.eq_policy_dict
    eq_endo_dict = model.eq_endo_dict
    eq_coeff_dict = model.eq_coeff_dict

    def analyse_eq(t, courant, num_eq):
        """
        Analyse une equation : 

        * déclaration des exogènes et des coefficients à la volée
        * construit les dictionnaires nom_var -> num_eq et num_eq -> nom_var

        """
        if t.data == 'define_eq':  # un seul signe = par équation
            leftpart, rightpart = t.children
            analyse_eq(leftpart, courant, num_eq)
            analyse_eq(rightpart, courant, num_eq)
            return

        elif t.data == 'add':
            leftpart, rightpart = t.children
            analyse_eq(leftpart, courant, num_eq)
            analyse_eq(rightpart, courant, num_eq)
            return

        elif t.data == 'sub':
            leftpart, rightpart = t.children
            analyse_eq(leftpart, courant, num_eq)
            analyse_eq(rightpart, courant, num_eq)
            return

        elif t.data == 'mul':
            leftpart, rightpart = t.children
            analyse_eq(leftpart, courant, num_eq)
            analyse_eq(rightpart, courant, num_eq)
            return

        elif t.data == 'div':
            leftpart, rightpart = t.children
            analyse_eq(leftpart, courant, num_eq)
            analyse_eq(rightpart, courant, num_eq)
            return

        elif t.data == 'pow':
            leftpart, rightpart = t.children
            analyse_eq(leftpart, courant, num_eq)
            analyse_eq(rightpart, courant, num_eq)
            return

        elif t.data == 'par':
            op = t.children[0]
            analyse_eq(op, courant, num_eq)
            return

        elif t.data == "delta":
            delay, part = t.children
            lag = int(delay)
            if lag > 0:
                analyse_eq(part, courant, num_eq)
                analyse_eq(part, courant + int(delay), num_eq)
                return
            else:
                raise ValueError("Illegal value for the lag !")

        elif t.data == "deltaone":  # pour prendre en compte l'ommission en cas de lag unitaire
            part = t.children[0]
            analyse_eq(part, courant, num_eq)
            analyse_eq(part, courant + 1, num_eq)
            return

        elif t.data == "lag":
            expr, delay = t.children
            lag = int(delay)
            if lag <= 0:
                analyse_eq(expr, courant + abs(lag), num_eq)
                return
            else:
                raise ValueError("Anticipated variables in the model !")

        elif t.data == "coeff":
            nom = str(t.children[0])
            if nom not in name_coeff_list:
                # on ajoute le coefficient déclaré à la volée
                name_coeff_list.append(nom)
            eq_coeff_dict[num_eq] = eq_coeff_dict[num_eq] | {nom}
            return

        elif t.data == "var":
            nom = str(t.children[0])

            if nom in name_exo_list:  # cas d'une variable exogène
                if nom not in var_eq_dict.keys():
                    var_eq_dict[nom] = {num_eq}
                else:
                    var_eq_dict[nom] = var_eq_dict[nom] | {
                        num_eq}  # Ajout pour décomposition D-M

                eq_exo_dict[num_eq] = eq_exo_dict[num_eq] | {nom}
                return

            elif nom in name_policy_list:  # cas d'une variable exogène
                if nom not in var_eq_dict.keys():
                    var_eq_dict[nom] = {num_eq}
                else:
                    var_eq_dict[nom] = var_eq_dict[nom] | {
                        num_eq}  # Ajout pour décomposition D-M

                eq_policy_dict[num_eq] = eq_policy_dict[num_eq] | {nom}
                return

            elif nom in name_endo_list:  # cas d'une variable endogène
                if nom not in var_eq_dict.keys():
                    var_eq_dict[nom] = {num_eq}
                else:
                    var_eq_dict[nom] = var_eq_dict[nom] | {
                        num_eq}  # Ajout pour décomposition D-M

                if courant == 0:  # variable dont la valeur doit être déterminée
                    eq_endo_dict[num_eq] = eq_endo_dict[num_eq] | {nom}

                return

            elif nom in name_coeff_list:
                eq_coeff_dict[num_eq] = eq_coeff_dict[num_eq] | {nom}
                return

            else:  # déclaration implicite d'une exogène
                name_exo_list.append(nom)
                var_eq_dict[nom] = {num_eq}
                eq_exo_dict[num_eq] = eq_exo_dict[num_eq] | {nom}
                return

        elif t.data == "log":
            op = t.children[0]
            analyse_eq(op, courant, num_eq)
            return

        elif t.data == "exp":
            op = t.children[0]
            analyse_eq(op, courant, num_eq)
            return

        elif t.data == 'neg':
            op = t.children[0]
            analyse_eq(op, courant, num_eq)
            return

        elif t.data == 'pos':
            op = t.children[0]
            analyse_eq(op, courant, num_eq)
            return

        elif (t.data == "number") or (t.data == "signednumber"):
            return

        elif t.data == "diff":
            return

        else:
            raise SyntaxError('Unknown instruction: %s' % t.data)

    start_time = time()    # tic

    # on boucle sur les équations

    for item in model.eq_obj_dict.keys():

        # on parse l'équation
        eq_parse = parser.parse(model.eq_obj_dict[item].text_eq)

        # unpack
        num_eq = model.eq_obj_dict[item].num_eq

        analyse_eq(eq_parse, 0, num_eq)       # analyse l'équation

        # mise à jour de l'arbre syntaxique dans l'objet équation
        model.eq_obj_dict[item].tree_eq = eq_parse

        model.eq_obj_dict[item].coeff_name_list = unique_list(
            eq_coeff_dict[num_eq])
        model.eq_obj_dict[item].exo_name_list = unique_list(
            eq_exo_dict[num_eq])
        model.eq_obj_dict[item].policy_name_list = unique_list(
            eq_policy_dict[num_eq])
        model.eq_obj_dict[item].endo_name_list = unique_list(
            eq_endo_dict[num_eq])

    model.name_endo_list = unique_list(
        name_endo_list)  # On a toutes les endogènes !

    model.name_exo_list = unique_list(
        name_exo_list)  # on a toutes les exogènes !

    model.name_policy_list = unique_list(name_policy_list)

    model.name_coeff_list = unique_list(
        name_coeff_list)  # on a tous les coefficients !

    model.coln_list = model.name_endo_list + model.name_exo_list + \
        model.name_policy_list  # on a toutes les variables du modèle !

    model.dicovar = {}
    for i in range(len(model.coln_list)):
        # crée un dictionnaire de correspondance globale nom / indice
        model.dicovar[model.coln_list[i]] = i

    model.dicoeff = {}
    for i in range(len(model.name_coeff_list)):
        # crée un dictionnaire  de correspondance  globale nom / indice
        model.dicoeff[model.name_coeff_list[i]] = i

    elapsed_time = time() - start_time  # toc

    print(f"The analysis of the model took {elapsed_time:.3f} secondes.\n")

    derive_model(model)

    for item in model.name_endo_list:
        model.symboles_dict[item] = "endogenous"

    for item in model.name_exo_list:
        model.symboles_dict[item] = "exogenous"

    for item in model.name_policy_list:
        model.symboles_dict[item] = "policy"

    for item in model.name_coeff_list:
        model.symboles_dict[item] = "coefficient"

    model.is_analyzed = True    # le modèle est désormais analysé

    return


def derive_equation(eq):
    """
    Compute the syntaxic trees for the derivatives
    """

    # unpack

    tree_eq = eq.tree_eq
    endo = eq.endo_name_list
    exo = eq.exo_name_list + eq.policy_name_list
    coeff = eq.coeff_name_list

    eq.tree_diff = deriv_tree(tree_eq, 0, endo, exo, coeff)


def derive_model(model, debug=False):
    """
    Compute the derivatives of the whole model
    """

    # unpack

    eq_obj_dict = model.eq_obj_dict
    dicoeff = model.dicoeff
    dicovar = model.dicovar

    start_time = time()    # tic

    for item in eq_obj_dict.keys():

        if debug:
            print(item)

        eq = eq_obj_dict[item]

        for c in eq.coeff_name_list:
            eq.coeff_eq_dict[c] = dicoeff[c]

        for ex in eq.exo_name_list:
            eq.exo_eq_dict[ex] = dicovar[ex]

        for en in eq.endo_name_list:
            eq.endo_eq_dict[en] = dicovar[en]

        for po in eq.policy_name_list:
            eq.policy_eq_dict[en] = dicovar[po]

        derive_equation(eq)

    elapsed_time = time() - start_time  # toc

    print(f"The computation of the derivatives of the model took {elapsed_time:.3f} secondes.\n")

    return


def deriv_tree(t, courant, endo, exo, coeff):
    """
    Calcul de l'arbre syntaxique dérivé d'une équation

    Version avec simplification de l'expression
    """

    if courant > 0:
        return Tree(data="number", children=[Token(type_="NUMBER", value="0")])

    if t.data == "define_eq":
        # a = b -> a' = b'
        leftpart, rightpart = t.children
        leftderiv = deriv_tree(leftpart, courant, endo, exo, coeff)
        rightderiv = deriv_tree(rightpart, courant, endo, exo, coeff)

        return Tree(data=t.data, children=[leftderiv, rightderiv])

    elif (t.data == 'add'):
        # a +/- b -> a' +/- b'
        leftpart, rightpart = t.children
        leftderiv = deriv_tree(leftpart, courant, endo, exo, coeff)
        rightderiv = deriv_tree(rightpart, courant, endo, exo, coeff)

        # simplification de l'addition

        if (leftderiv.data == "number") and (leftderiv.children[0].value == "0"):
            # a' = 0
            if (rightderiv.data == "number") and (rightderiv.children[0].value == "0"):
                # b' = 0
                return Tree(data="number", children=[Token(type_="NUMBER", value="0")])
            else:
                return rightderiv
        else:
            if (rightderiv.data == "number") and (rightderiv.children[0].value == "0"):
                # b' = 0
                return leftderiv
            else:
                return Tree(data=t.data, children=[leftderiv, rightderiv])

    elif (t.data == "sub"):
        # a +/- b -> a' +/- b'
        leftpart, rightpart = t.children
        leftderiv = deriv_tree(leftpart, courant, endo, exo, coeff)
        rightderiv = deriv_tree(rightpart, courant, endo, exo, coeff)
        # simplification de la soustraction

        if (leftderiv.data == "number") and (leftderiv.children[0].value == "0"):
            # a' = 0
            if (rightderiv.data == "number") and (rightderiv.children[0].value == "0"):
                # b' = 0
                return Tree(data="number", children=[Token(type_="NUMBER", value="0")])
            else:
                return Tree(data="neg", children=[rightderiv])
        else:
            if (rightderiv.data == "number") and (rightderiv.children[0].value == "0"):
                # b' = 0
                return leftderiv
            else:
                return Tree(data=t.data, children=[leftderiv, rightderiv])

    elif t.data == 'mul':
        # a * b -> a' * b + a * b'
        leftpart, rightpart = t.children
        leftderiv = deriv_tree(leftpart, courant, endo, exo, coeff)
        rightderiv = deriv_tree(rightpart, courant, endo, exo, coeff)
        # simplification du produit
        if (leftderiv.data == "number") and (leftderiv.children[0].value == "0"):
            # a' = 0
            if (rightderiv.data == "number") and (rightderiv.children[0].value == "0"):
                # b' = 0
                return Tree(data="number", children=[Token(type_="NUMBER", value="0")])
            else:
                rhs = Tree(data="mul", children=[leftpart, rightderiv])
                return rhs
        else:
            if (rightderiv.data == "number") and (rightderiv.children[0].value == "0"):
                # b' = 0
                lhs = Tree(data="mul", children=[leftderiv, rightpart])
                return lhs
            else:
                lhs = Tree(data="mul", children=[leftderiv, rightpart])
                rhs = Tree(data="mul", children=[leftpart, rightderiv])
                return Tree(data="add", children=[lhs, rhs])

    elif t.data == "div":
        # a / b -> a' / b - a * b' / b^2
        leftpart, rightpart = t.children
        leftderiv = deriv_tree(leftpart, courant, endo, exo, coeff)
        rightderiv = deriv_tree(rightpart, courant, endo, exo, coeff)
        # simplification de la division
        if (leftderiv.data == "number") and (leftderiv.children[0].value == "0"):
            # a' = 0
            if (rightderiv.data == "number") and (rightderiv.children[0].value == "0"):
                # b' = 0
                return Tree(data="number", children=[Token(type_="NUMBER", value="0")])
            else:
                mhs1 = Tree(data="div", children=[leftpart, rightpart])
                mhs2 = Tree(data="div", children=[rightderiv, rightpart])
                rhs = Tree(data="mul", children=[mhs1, mhs2])
                return Tree(data="neg", children=[rhs])
        else:
            if (rightderiv.data == "number") and (rightderiv.children[0].value == "0"):
                # b' = 0
                lhs = Tree(data="div", children=[leftderiv, rightpart])
                return lhs
            else:
                lhs = Tree(data="div", children=[leftderiv, rightpart])
                mhs1 = Tree(data="div", children=[leftpart, rightpart])
                mhs2 = Tree(data="div", children=[rightderiv, rightpart])
                rhs = Tree(data="mul", children=[mhs1, mhs2])
                return Tree(data="sub", children=[lhs, rhs])

    elif t.data == "pow":        # pas d'endogène dans la puissance !!!
        #  a^b -> a' * b * a^(b-1)
        leftpart, rightpart = t.children  # rightpart -> la puissance
        leftderiv = deriv_tree(leftpart, courant, endo, exo, coeff)  # a'
        if (leftderiv.data == "number") and (leftderiv.children[0].value == "0"):
            # a' = 0
            return Tree(data="number", children=[Token(type_="NUMBER", value="0")])
        else:
            newpow = Tree(data="sub", children=[rightpart, Tree(
                data="number", children=[Token(type_="NUMBER", value='1')])])
            rhs = Tree(data="pow", children=[leftpart, newpow])  # a^(b-1)
            mhs = Tree(data="mul", children=[rightpart, rhs])
            return Tree(data="mul", children=[leftderiv, mhs])

    elif t.data == "delta":
        # delta(lag:op) -> (op')
        delay, part = t.children

        op = deriv_tree(part, courant, endo, exo, coeff)

        return op

    elif t.data == "deltaone":
        # delta(op) -> (op')
        op = deriv_tree(t.children[0], courant, endo, exo, coeff)

        return op

    elif t.data == "lag":
        # a(-lag)
        expr, delay = t.children
        lag = int(delay)
        if lag == 0:
            op = deriv_tree(expr, courant + abs(lag), endo, exo, coeff)
            return op
        else:
            return Tree(data="number", children=[Token(type_="NUMBER", value='0')])

    elif t.data == "coeff":
        op = Token(type_="NUMBER", value="0")    # coeff' = 0
        return Tree(data="number", children=[op])

    elif (t.data == "number") or (t.data == "signednumber"):
        op = Token(type_="NUMBER", value="0")    # number' = 0
        return Tree(data=t.data, children=[op])

    elif (t.data == 'neg') or (t.data == 'pos') or (t.data == "par"):
        op = deriv_tree(t.children[0], courant, endo, exo, coeff)
        if (op.data == "number") and (op.children[0].value == "0"):
            return Tree(data="number", children=[Token(type_="NUMBER", value="0")])
        else:
            return Tree(data=t.data, children=[op])

    elif t.data == "log":
        # log(a) -> a' / a
        op = deriv_tree(t.children[0], courant, endo, exo, coeff)
        if (op.data == "number") and (op.children[0].value == "0"):
            return Tree(data="number", children=[Token(type_="NUMBER", value="0")])
        else:
            return Tree(data="div", children=[op, t.children[0]])

    elif t.data == "exp":
        # exp(a) -> a' * exp(a)
        op = deriv_tree(t.children[0], courant, endo, exo, coeff)
        if (op.data == "number") and (op.children[0].value == "0"):
            return Tree(data="number", children=[Token(type_="NUMBER", value="0")])
        else:
            rightpart = Tree(data="exp", children=[t.children[0]])
            return Tree(data="mul", children=[op, rightpart])

    elif t.data == "var":  # !!! cas d'une variable !!!
        nom = str(t.children[0])
        if (nom in exo) or (nom in coeff):  # cas d'une variable exogène ou d'un coefficient
            op = Token(type_="NUMBER", value="0")
            return Tree(data="number", children=[op])    # exo' = 0

        elif nom in endo:  # cas d'une variable endogène

            if courant == 0:  # On introduit dx qui vaudra 0 ou 1 lors de l'évaluation
                return Tree(data="diff", children=t.children)

            else:  # L'endogène est retardée
                op = Token(type_="NUMBER", value="0")
                return Tree(data="number", children=[op])

        else:
            raise SyntaxError('Unknown variable')

    else:
        raise SyntaxError('Unknown instruction: %s' % t.data)


def write_yaml_file(model, yaml_filename,  dir="./modeles_python"):
    """
    Write all the information about the model in a yaml file

    yaml_filename : name of the yaml file (with .yaml extension)
    """

    if (not model.is_analyzed):

        raise ValueError("The model is not analyzed.")

    mod_dict = [{'name_mod': model.name_mod}, {'name_endo_list': model.name_endo_list},
                {'name_exo_list': model.name_exo_list}, {
                    'name_policy_list': model.name_policy_list},
                {'name_coeff_list': model.name_coeff_list}]

    mod_eq = dict()

    for item in model.eq_obj_dict.keys():
        mod_eq[item] = {'name_eq': model.eq_obj_dict[item].name_eq, 'text_eq': model.eq_obj_dict[item].text_eq, 'num_eq':  model.eq_obj_dict[item].num_eq,
                        'coeff_eq_dict': model.eq_obj_dict[item].coeff_eq_dict, 'coeff_name_list': model.eq_obj_dict[item].coeff_name_list,
                        'endo_eq_dict': model.eq_obj_dict[item].endo_eq_dict, 'endo_name_list': model.eq_obj_dict[item].endo_name_list,
                        'exo_eq_dict': model.eq_obj_dict[item].exo_eq_dict, 'exo_name_list': model.eq_obj_dict[item].exo_name_list,
                        'policy_eq_dict': model.eq_obj_dict[item].policy_eq_dict, 'policy_name_list': model.eq_obj_dict[item].policy_name_list}

    with open(dir+"/"+yaml_filename, 'w+') as f:

        dump(mod_dict, f)
        dump([{'equations': mod_eq}], f)

    return
