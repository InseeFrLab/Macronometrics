# -*- coding: utf-8 -*-
"""
@author: QME8JI
"""


class Block():
    """
    Classe définissant les outils de construction d'un bloc d'un modèle.

    Le modèle doit être analysé auparavant avec les outils de la classe Analyse.

    ATTENTION : pour le moment, on considère le texte des équations comme étant
    syntaxiquement correct (au sens de Troll)
    """

    def __init__(self, model, list_eq_block, list_endo_block, list_exo_block, n_block):
        """
        Constructeur de classe Symbolic

        Permet la traduction d'un bloc du modèle

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
        self.list_eq_block = list_eq_block
        self.list_endo_block = list_endo_block
        self.list_exo_block = list_exo_block

        self.model = model

        self.block_eq_obj_list = [self.model.eq_obj_dict_number[i]
                                  for i in self.list_eq_block]

        self.n_block = n_block

    def translate_block_python(self):
        """ 
        Permet la traduction d'un  modèle en python.
        Le modèle doit être analysé avec les outils de la bibliothèque Analyse.

        Evalue l'ensemble des équations
        -> Construit le corps du texte d'une fonction f des endogènes dont la valeur cible est
        solution de f(x) = 0 sous forme de chaine de caractères
        -> Construit également la jacobienne 

        ATTENTION : pour le moment, on considère le texte des équations comme étant
                syntaxiquement correct (au sens de Troll)

        Paramètres
        ==========

        model : un modèle de la classe Modele préalablement analysé

        """

        # unpack
        name_coeff_list = self.model.name_coeff_list
        dicovar = self.model.dicovar

        def run_instruction(t, courant, vari=None):
            # """
            # Règles de production (evaluation) pour une équation.

            # On suppose que le modèle est correctement analysé.

            # Le but est de produire une chaine de caractères (à passer à la fonction eval de Python
            # ou à enregistrer dans une fonction).

            # Arguments :
            # * t : arbre syntaxique
            # * courant : date courante (pour les retards)
            #     -> on démarre à courant = 0 et on doit disposer des endogènes jusqu'à l'instant précédent.
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
                if nom not in name_coeff_list:
                    raise ValueError("Valeur inconnue pour le coefficient !")
                # on va chercher le coefficient dans un dictionnaire
                return "coeff['" + nom + "']"

            elif t.data == "var":
                nom = str(t.children[0])

                # cas d'une variable exogène ou policy
                if (nom in self.list_exo_block):
                    # on cherche dans la base de données du modèle la valeur
                    return "data[t-"+str(courant)+"," + str(dicovar[nom]) + "]"

                elif nom in self.list_endo_block:  # cas d'une variable endogène

                    if courant == 0:  # variable dont la valeur doit être déterminée

                        return "x["+str(self.list_endo_block.index(nom))+"]"

                    else:  # on cherche dans la base de données du modèle
                        return "data[t-"+str(courant)+"," + str(dicovar[nom]) + "]"

                elif nom in name_coeff_list:
                    # on va chercher le coefficient dans un dictionnaire
                    return "coeff['" + nom + "']"

                else:
                    raise ValueError("Le modèle doit être analysé !")

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

        dimbloc = str(len(self.list_endo_block))

        res_block = "\tf = np.zeros("+dimbloc+",dtype=np.float64)\n"
        jac_block = "\tdf = np.zeros(("+dimbloc+","+dimbloc+"),dtype=np.float64)\n"

        ell = 0 

        for eq in self.block_eq_obj_list:

            # on récupère l'arbre syntaxique de l'équation et de sa dérivée
            eq_parse = eq.tree_eq
            jac_parse = eq.tree_diff

            texte_eq = run_instruction(eq_parse, 0)

            res_block += "\tf[" + str(ell) + "] = "
            res_block += texte_eq + "\n"  # on met à jour le texte de la fonction

            # ensemble des endogènes contemporaines de l'équation
            endo_name_list = eq.endo_name_list

            for j, vari in enumerate(self.list_endo_block):

                

                if vari in endo_name_list:
                    jac_block += "\tdf[" + str(ell) + "][" + str(j) + "] = "
                    partialder = run_instruction(jac_parse, 0, vari)
                    jac_block += partialder + "\n"

            ell += 1

        self.fun_text = res_block
        self.jac_text = jac_block


        return
