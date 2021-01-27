# -*- coding: utf-8 -*-
"""
@author: QME8JI
"""
import pandas as pd

def unique_list(liste):
    """
    Return a list without duplicate entries.
    """
    return list(set(liste))


class Estim():
    """
    Classe permettant l'estimation d'une équation.

    Le modèle doit être analysé auparavant avec les outils de la classe Analyse.

    ATTENTION : pour le moment, on considère le texte des équations comme étant
    syntaxiquement correct (au sens de Troll)
    """

    def __init__(self, equation, model, df_mod):
        """


        Paramètres
        ==========

        list_eq_block : ordre des équations dans le block
        set_eq : ensemble d'arbres syntaxiques
        set_diff : ensemble d'arbres dérivés       
        dict_eq_var : dictionnaire de correspondance equation -> nom de variable
        endo : nom des endogènes 
        exo : nom des exogènes et des policy issues de la lecture du modèle
        dicovar : correspondance variable -> indice globale (structure de données)
        coeff : nom des coefficients issus de la lecture du modèle 
        dicoeff : correspondance coefficient -> indice

        """
        self.equation = equation

        self.model = model

        self.coeff_name_list = equation.coeff_name_list

        self.coeff_eq_dict_loc = dict()

        self.fun_text = ""

        self.n_coeff = len(self.coeff_name_list)

        self.var_list_loc = unique_list(equation.policy_name_list 
                                        + equation.endo_lag_name_list 
                                        + equation.endo_name_list 
                                        + equation.exo_name_list)

        self.var_list_loc.sort()

        self.var_eq_dict_loc = dict()

        if len(self.var_list_loc) ==0 :
            print("Pas de variable dans l'équation !\n")
        else:
            print("Variables dans l'équation\n")
            print(self.var_list_loc)
            for i,v in enumerate(self.var_list_loc):
                self.var_eq_dict_loc[v] = i


        if self.n_coeff ==0 :
            print("Pas de coefficient à estimer !\n")
        else:
            print(str(self.n_coeff)+" coefficient(s) à estimer\n")
            print(self.coeff_name_list)
            for i,v in enumerate(self.coeff_name_list):
                self.coeff_eq_dict_loc[v] = i

        self.data_eq = df_mod[self.var_list_loc].copy()


    def create_estimfun_python(self):
        """ 
        Permet la traduction d'une  équation en python pour l'estimation de ses coefficients.
        Le modèle doit être analysé avec les outils de la bibliothèque Analyse.

        ATTENTION : pour le moment, on considère le texte des équations comme étant
                syntaxiquement correct (au sens de Troll)

        Paramètres
        ==========

        model : un modèle de la classe Modele préalablement analysé

        """
        dicovar = self.var_eq_dict_loc        

        def run_instruction(t, courant, vari=None):
            # """
            # Règles de production (evaluation) pour une équation.

            # On suppose que le modèle est correctement analysé.

            # Le but est de produire une chaine de caractères (à passer à la fonction eval de Python
            # ou à enregistrer dans une fonction).

            # Arguments :
            # * t : arbre syntaxique
            # * courant : date courante (pour les retards)
            #     -> on démarre à courant = 0 et on doit disposer des endogènes jusqu'à l'instant donné.
            # * vari : nom de variable (pour les dérivées partielles)
            # """

            if t.data == 'define_eq':  # un seul signe = par équation
                leftpart, rightpart = t.children

                t1 = run_instruction(leftpart, courant, vari)
                t2 = run_instruction(rightpart, courant, vari)

                if t1 == "0":
                    if t2 == "0":
                        return "0"
                    elif rightpart.data == 'par':
                        return "-"+t2
                    else:
                        return "-("+t2+")"
                elif t2 == "0":
                    if leftpart.data == 'par':
                        return t1
                    else:
                        return '('+t1+')'
                elif rightpart.data == 'par':
                    if leftpart.data == 'par':
                        return t1+'-'+t2
                    else:
                        return '('+t1+')-'+t2
                elif leftpart.data == "par":
                    return t1+'-('+t2+')'
                else:
                    return '('+t1+')-('+t2+')'

            elif t.data == 'add':
                leftpart, rightpart = t.children
                t1 = run_instruction(leftpart, courant, vari)
                t2 = run_instruction(rightpart, courant, vari)

                if t1 == "0":
                    if t2 == "0":
                        return "0"
                    else:
                        return t2
                elif t2 == "0":
                    return t1
                else:
                    if leftpart.children == 'par':
                        if rightpart.children == 'par':
                            return t1+'+'+t2
                        else:
                            return t1 + '('+t2+')'
                    elif rightpart.children == 'par':
                        return '('+t1+')'+t2
                    else:
                        return '('+t1+')+('+t2+')'

            elif t.data == 'sub':
                leftpart, rightpart = t.children
                t1 = run_instruction(leftpart, courant, vari)
                t2 = run_instruction(rightpart, courant, vari)

                if t1 == "0":
                    if t2 == "0":
                        return "0"
                    elif rightpart.data == 'par':
                        return "-"+t2
                    else:
                        return "-("+t2+")"
                elif t2 == "0":
                    if leftpart.data == 'par':
                        return t1
                    else:
                        return '('+t1+')'
                elif rightpart.data == 'par':
                    if leftpart.data == 'par':
                        return t1+'-'+t2
                    else:
                        return '('+t1+')-'+t2
                elif leftpart.data == "par":
                    return t1+'-('+t2+')'
                else:
                    return '('+t1+')-('+t2+')'

            elif t.data == 'mul':
                leftpart, rightpart = t.children
                t1 = run_instruction(leftpart, courant, vari)
                t2 = run_instruction(rightpart, courant, vari)
                if t1 == "0" or t2 == "0":
                    return "0"
                elif t1 == "1":
                    return t2
                elif t2 == "1":
                    return t1
                else:
                    return "(" + t1 + ")*(" + t2 + ")"

            elif t.data == 'div':
                leftpart, rightpart = t.children
                t1 = run_instruction(leftpart, courant, vari)
                if t1 == "0":
                    return "0"
                elif t1 == "1":
                    return "1/(" + run_instruction(rightpart, courant, vari)+")"
                else:
                    return "("+t1+")" + "/(" + run_instruction(rightpart, courant, vari)+")"

            elif t.data == 'pow':
                leftpart, rightpart = t.children
                t1 = run_instruction(leftpart, courant, vari)
                if t1 == "0":
                    return "0"
                elif t1 == "1":
                    return "1"
                else:
                    return "(" + run_instruction(leftpart, courant, vari)+")" + "**" + "("+run_instruction(rightpart, courant, vari)+")"

            elif t.data == 'par':
                op = t.children[0]
                t1 = run_instruction(op, courant, vari)

                if t1 == "0":
                    return "0"
                elif t1 == "1":
                    return "1"
                elif op.data == 'par' or op.data == 'pos' or op.data == 'number' or op.data == "diff":
                    return t1
                else:
                    return "(" + t1 + ")"

            elif t.data == "delta":
                delay, part = t.children
                lag = int(delay)
                if lag > 0:
                    return "(" + run_instruction(part, courant, vari) + "- (" + run_instruction(part, courant + int(delay), vari) + "))"
                else:
                    raise ValueError("Valeur incohérente du retard !")

            elif t.data == "deltaone":  # pour prendre en compte l'ommission en cas de lag unitaire
                part = t.children[0]
                return "(" + run_instruction(part, courant, vari) + "- (" + run_instruction(part, courant + 1, vari) + "))"

            elif t.data == "lag":
                expr, delay = t.children
                lag = int(delay)
                if lag <= 0:
                    return run_instruction(expr, courant + abs(lag), vari)
                else:
                    raise ValueError(
                        "Le modèle contient des variables anticipées !")

            elif t.data == "coeff":
                nom = str(t.children[0])
                if nom not in self.coeff_name_list:
                    raise ValueError("Valeur inconnue pour le coefficient !")
                # on va chercher le coefficient dans un dictionnaire
                return "_z[" + str(self.coeff_eq_dict_loc[nom]) + "]"

            elif t.data == "var":
                nom = str(t.children[0])

                # cas d'une variable exogène ou policy
                # if (nom in self.list_exo_block):
                    # on cherche dans la base de données du modèle la valeur
                return "_data[_t-"+str(courant)+"," + str(dicovar[nom]) + "]"

                # elif nom in self.list_endo_block:  # cas d'une variable endogène

                #     if courant == 0:  # variable dont la valeur doit être déterminée

                #         return "x["+str(self.list_endo_block.index(nom))+"]"

                #     else:  # on cherche dans la base de données du modèle
                #         return "data[t-"+str(courant)+"," + str(dicovar[nom]) + "]"

                # elif nom in name_coeff_list:
                #     # on va chercher le coefficient dans un dictionnaire
                #     return "coeff['" + nom + "']"

                # else:
                #     raise ValueError("Le modèle doit être analysé !")

            elif t.data == "log":
                op = t.children[0]
                return "log(" + run_instruction(op, courant, vari) + ")"

            elif t.data == "exp":
                op = t.children[0]
                return "exp(" + run_instruction(op, courant, vari) + ")"

            elif t.data == 'neg':
                op = t.children[0]

                t1 = run_instruction(op, courant, vari)

                if t1 == "0":
                    return "0"
                elif t1 == "1":
                    return "-1"
                elif op.data == 'par' or op.data == 'pos' or op.data == 'number' or op.data == "diff":
                    return "-"+t1
                else:
                    return "-(" + t1 + ")"

            elif t.data == 'pos':
                op = t.children[0]
                return run_instruction(op, courant, vari)

            elif (t.data == "number") or (t.data == "signednumber"):
                valeur = t.children[0]
                if valeur == 0:
                    return "0"
                else:
                    return str(valeur)

            elif t.data == "diff":
                nom = str(t.children[0])
                if nom == vari:
                    return "1"
                else:
                    return "0"

            else:
                raise SyntaxError('Unknown instruction: %s' % t.data)



            # on récupère l'arbre syntaxique de l'équation et de sa dérivée
        eq_parse = self.equation.tree_eq


        texte_eq = run_instruction(eq_parse, 0)

        res_block = "\t_res = 0\n"
        res_block += "\tfor _t in range(_t_start,_t_stop):\n"
        res_block += "\t\t_res += ("
        res_block += texte_eq   # on met à jour le texte de la fonction
        res_block += ")**2\n"


            # # ensemble des endogènes contemporaines de l'équation
            # endo_name_list = eq.endo_name_list

            # for j, vari in enumerate(self.list_endo_block):

                

            #     if vari in endo_name_list:
            #         jac_block += "\tdf[" + str(ell) + "][" + str(j) + "] = "
            #         partialder = run_instruction(jac_parse, 0, vari)
            #         jac_block += partialder + "\n"

            # ell += 1

        self.fun_text = res_block



        return