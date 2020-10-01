# -*- coding: utf-8 -*-
"""
@author: QME8JI
"""
import numpy as np
import scipy as sp
import scipy.optimize as spo
# from numba import jit


def ggn11(xinit,fun,jac,args,ftol=1e-10,itermax=1000,alphamin=0.05) :
    """
    Implémentation de la variante de la méthode de Newton
    Grau, Grau-Sanchez et Noguera 2011
    x0 : valeur initiale 
    fun : fonction 
    jac : jacobienne
    args : autres arguments de la fonction 
    ftol : critère de convergence
    itermax : limite d'appels à la fonction
    alphamin : paramètre de relaxation
    """

    x0 = xinit # point courant
    f0 = fun(x0,*args) # évaluation de la fonction au point courant
    maxf0 = np.linalg.norm(f0,np.inf) # norme infinie de f(x0)

    decreasing = True # pour un critère de décroissance globale
    iterate = 0

    while (maxf0 > ftol) & decreasing & (iterate < itermax) :
        iterate = iterate + 1
        
        J0 = jac(x0,*args) # F'(x0) jacobienne au point courant
        IJ0f0 =  np.linalg.inv(J0) @ f0 # F'(x0)^{-1} * F(x0)
        y1 = x0 - IJ0f0 # Etape de l'algorithme de Newton usuel

        Jy1 = jac(y1,*args) # F'(y1)
        IJy1 = np.linalg.inv(Jy1)
        IJy1f0 = IJy1 @ f0  # F'(y1)^{-1} * F(x0)
        z1 = x0 - 0.5*(IJ0f0 + IJy1f0) # méthode de Newton harmonique

        fz1 = fun(z1,*args) # F(z1)
        IJy1fz1 = IJy1 @ fz1 # F'(y1)^{-1} * F(z1)
        x1 = z1 - IJy1fz1 # Méthode de Newton modifiée d'ordre 5 
        
        try :

            f0 = fun(x1,*args)
            maxf1 = np.linalg.norm(f0,np.inf)

            # mise à jour
            if (maxf1 < maxf0) :
                maxf0 = maxf1
                x0 = x1

            else :
                maxf1 = np.linalg.norm(fz1,np.inf) 

                if (maxf1 < maxf0) :
                    maxf0 = maxf1
                    x0 = z1
                    f0 = fz1 # on se replie sur la méthode harmonique

                else:
                    fy1 = fun(y1,*args)
                    maxf1 = np.linalg.norm(fy1,np.inf)

                    if (maxf1 < maxf0) : 
                        maxf0 = maxf1
                        x0 = y1
                        f0 = fy1 # on se replie sur la méthode de Newton 

                    else :
                        decreasing = False

        except :
            
            # relaxation / algorithme de Bank - Rose (1981)
            alpha = 0.9

            while (alpha > alphamin) :

                try :

                    x1 = x0 - alpha * IJ0f0 # méthode de Newton
                    f1 = fun(x1,*args)
                    alpha = alphamin

                except :

                    alpha = 0.5* alpha
            
            maxf1 = np.linalg.norm(f1,np.inf)

            if (maxf1 < maxf0)  :
                maxf0 = maxf1
                x0 = x1
                f0 = f1
            
            else :
                decreasing = False


    return (x0,f0,maxf0)

def newton(xinit,fun,jac,args,ftol=1e-10,itermax=1000,alphamin=0.05) :
    """
    Implémentation de la  méthode de Newton (avec relaxation)
    x0 : valeur initiale 
    fun : fonction 
    jac : jacobienne
    args : autres arguments de la fonction 
    ftol : critère de convergence
    itermax : limite d'appels à la fonction
    alphamin : paramètre de relaxation
    """

    x0 = xinit # point courant
    f0 = fun(x0,*args) # évaluation de la fonction au point courant
    maxf0 = np.linalg.norm(f0,np.inf) # norme infinie de f(x0)

    decreasing = True # pour un critère de décroissance globale
    iterate = 0

    while (maxf0 > ftol) & decreasing & (iterate < itermax) :
        iterate = iterate + 1
        
        J0 = jac(x0,*args) # F'(x0) jacobienne au point courant
        IJ0f0 =  np.linalg.inv(J0) @ f0 # F'(x0)^{-1} * F(x0)
        y1 = x0 - IJ0f0 # Etape de l'algorithme de Newton usuel

        
        try :

 
            fy1 = fun(y1,*args)
            maxf1 = np.linalg.norm(fy1,np.inf)

            if (maxf1 < maxf0) : 
                maxf0 = maxf1
                x0 = y1
                f0 = fy1 # on se replie sur la méthode de Newton 

            else :
                decreasing = False

        except :
            
            # relaxation / algorithme de Bank - Rose (1981)
            alpha = 0.9

            while (alpha > alphamin) :

                try :

                    x1 = x0 - alpha * IJ0f0 # méthode de Newton
                    f1 = fun(x1,*args)
                    alpha = alphamin

                except :

                    alpha = 0.5* alpha
            
            maxf1 = np.linalg.norm(f1,np.inf)

            if (maxf1 < maxf0)  :
                maxf0 = maxf1
                x0 = x1
                f0 = f1
            
            else :
                decreasing = False

    return (x0,f0,maxf0)

def sp_root(xinit,fun,jac,args,ftol=1e-10) :
    """
    Encapsulation de la méthode standard root de scipy
    x0 : valeur initiale 
    fun : fonction 
    jac : jacobienne
    args : autres arguments de la fonction 
    ftol : critère de convergence
    itermax : limite d'appels à la fonction
    alphamin : paramètre de relaxation
    """

    res_spo = spo.root(fun,xinit,args,jac=jac,tol=ftol) 
    return (res_spo.x, res_spo.fun, np.linalg.norm(res_spo.fun,np.inf))

def newton_alt(xinit,fun,jac,args,ftol=1e-10,itermax=1000,alphamin=0.05) :
    """
    Implémentation de la  méthode de Newton (avec relaxation)
    Changement dans le calcul des inverses matricielles
    x0 : valeur initiale 
    fun : fonction 
    jac : jacobienne
    args : autres arguments de la fonction 
    ftol : critère de convergence
    itermax : limite d'appels à la fonction
    alphamin : paramètre de relaxation
    """

    x0 = xinit # point courant
    f0 = fun(x0,*args) # évaluation de la fonction au point courant
    maxf0 = np.linalg.norm(f0,np.inf) # norme infinie de f(x0)

    decreasing = True # pour un critère de décroissance globale
    iterate = 0

    while (maxf0 > ftol) & decreasing & (iterate < itermax) :
        iterate = iterate + 1
        
        J0 = jac(x0,*args) # F'(x0) jacobienne au point courant
        IJ0f0 =  np.linalg.solve(J0, f0) # F'(x0)^{-1} * F(x0)
        y1 = x0 - IJ0f0 # Etape de l'algorithme de Newton usuel

        # Jy1 = jac(y1,*args) # F'(y1)
        # IJy1 = np.linalg.inv(Jy1)
        # IJy1f0 = IJy1 @ f0  # F'(y1)^{-1} * F(x0)
        # z1 = x0 - 0.5*(IJ0f0 + IJy1f0) # méthode de Newton harmonique

        # fz1 = fun(z1,*args) # F(z1)
        # IJy1fz1 = IJy1 @ fz1 # F'(y1)^{-1} * F(z1)
        # x1 = z1 - IJy1fz1 # Méthode de Newton modifiée d'ordre 5 
        
        try :

            # f0 = fun(x1,*args)
            # maxf1 = np.linalg.norm(f0,np.inf)

            # # mise à jour
            # if (maxf1 < maxf0) :
            #     maxf0 = maxf1
            #     x0 = x1

            # else :
            #     maxf1 = np.linalg.norm(fz1,np.inf) 

            #     if (maxf1 < maxf0) :
            #         maxf0 = maxf1
            #         x0 = z1
            #         f0 = fz1 # on se replie sur la méthode harmonique

            #     else:
            fy1 = fun(y1,*args)
            maxf1 = np.linalg.norm(fy1,np.inf)

            if (maxf1 < maxf0) : 
                maxf0 = maxf1
                x0 = y1
                f0 = fy1 # on se replie sur la méthode de Newton 

            else :
                decreasing = False

        except :
            
            # relaxation / algorithme de Bank - Rose (1981)
            alpha = 0.9

            while (alpha > alphamin) :

                try :

                    x1 = x0 - alpha * IJ0f0 # méthode de Newton
                    f1 = fun(x1,*args)
                    alpha = alphamin

                except :

                    alpha = 0.5* alpha
            
            maxf1 = np.linalg.norm(f1,np.inf)

            if (maxf1 < maxf0)  :
                maxf0 = maxf1
                x0 = x1
                f0 = f1
            
            else :
                decreasing = False


    return (x0,f0,maxf0)

# @jit(nopython=True)
# def newton_numba(xinit,fun,jac,args,ftol=1e-10,itermax=1000,alphamin=0.05) :
#     """
#     Implémentation de la  méthode de Newton (avec relaxation)
#     Changement dans le calcul des inverses matricielles
#     x0 : valeur initiale 
#     fun : fonction 
#     jac : jacobienne
#     args : autres arguments de la fonction 
#     ftol : critère de convergence
#     itermax : limite d'appels à la fonction
#     alphamin : paramètre de relaxation
#     """

#     x0 = xinit # point courant
#     f0 = fun(x0,*args) # évaluation de la fonction au point courant
#     maxf0 = np.linalg.norm(f0,np.inf) # norme infinie de f(x0)

#     decreasing = True # pour un critère de décroissance globale
#     iterate = 0

#     while (maxf0 > ftol) & decreasing & (iterate < itermax) :
#         iterate = iterate + 1
        
#         J0 = jac(x0,*args) # F'(x0) jacobienne au point courant
#         IJ0f0 =  np.linalg.solve(J0, f0) # F'(x0)^{-1} * F(x0)
#         y1 = x0 - IJ0f0 # Etape de l'algorithme de Newton usuel

        
#         try :

#             fy1 = fun(y1,*args)
#             maxf1 = np.linalg.norm(fy1,np.inf)

#             if (maxf1 < maxf0) : 
#                 maxf0 = maxf1
#                 x0 = y1
#                 f0 = fy1 # on se replie sur la méthode de Newton 

#             else :
#                 decreasing = False

#         except :
            
#             # relaxation / algorithme de Bank - Rose (1981)
#             alpha = 0.9

#             while (alpha > alphamin) :

#                 try :

#                     x1 = x0 - alpha * IJ0f0 # méthode de Newton
#                     f1 = fun(x1,*args)
#                     alpha = alphamin

#                 except :

#                     alpha = 0.5* alpha
            
#             maxf1 = np.linalg.norm(f1,np.inf)

#             if (maxf1 < maxf0)  :
#                 maxf0 = maxf1
#                 x0 = x1
#                 f0 = f1
            
#             else :
#                 decreasing = False


#     return (x0,f0,maxf0)