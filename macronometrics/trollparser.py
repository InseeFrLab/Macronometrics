# -*- coding: utf-8 -*-
"""
@author: QME8JI
"""

from lark import Lark # Parseur générique

#definition de la grammaire
equation_grammar = '''
                ?start: sum
                    | sum "=" sum    -> define_eq
                     
                ?sum: product
                    | sum "+" product   -> add
                    | sum "-" product   -> sub

                ?product: power
                    | product "*" power  -> mul
                    | product "/" power  -> div
                     
                ?power: unary
                    | power "^" unary -> pow 
                     
                ?unary: item
                    | "log(" sum  ")"     -> log
                    | "exp(" sum  ")"     -> exp
                    | "-" unary         -> neg
                    | "+" unary -> pos
                    | "(" sum ")"     -> par
                    | "del(" NUMBER ":" sum ")" -> delta 
                    | "del(" sum ")" -> deltaone
                    | unary "(" SIGNED_NUMBER ")" -> lag
                     
                ?item: NUMBER -> number
                    | SIGNED_NUMBER -> signednumber
                    | CNAME"'c" -> coeff
                    | CNAME -> var

                %import common.CNAME    
                %import common.NUMBER
                %import common.SIGNED_NUMBER
                %import common.WS
                %ignore WS
                '''
        

# Définition du parseur Troll
parser = Lark(equation_grammar, start="start")
