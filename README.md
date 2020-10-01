# Macronometrics

Version 0.0.1

## A toolbox for macroeconometric modeling :

 * High-level language for model description (parser based on Lark)
 * backward looking modeling with AR / ECM processes
 * Dulmage - Mendelsohn block decomposition of the model
 * Symbolic computation of the jacobian 
 * Several choices of numerical solvers (based on Scipy, or high-order Newton methods)
 * Time-series management based on Pandas

## To do :

 * Numba just-in-time compilation of the solving functions
 * Estimation of the coefficients of the model (OLS)
 * Cython compilation of the solving functions

## Acknowledgements :

The code for Dulmage - Mendelsohn block decomposition is implemented with courtesy of Bank of Japan research team :

Hirakata, N., K. Kazutoshi, A. Kanafuji, Y. Kido, Y. Kishaba, T. Murakoshi, and T. Shinohara (2019) "The Quarterly Japanese Economic Model (Q-JEM): 2019 version" Bank of Japan Working Paper Series, No. 19-E-7.

Some features of the toolbox are inspired from the Grocer package for Scilab, and implemented with courtesy of Eric Dubois, lead developer of Grocer : http://dubois.ensae.net/grocer.html 

## Authors :

Institut National de la Statistique et des Etudes Economiques  
Direction des Etudes et Synthèses Economiques  
Département des Etudes Economiques  
Division des Etudes Macroéconomiques

Benjamin Favetto - Adrien Lagouge - Olivier Simon




