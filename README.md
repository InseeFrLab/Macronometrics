# Macronometrics

A toolbox for macroeconometric modeling 

Version 0.0.1

## Key features

 * High-level language for model description (parser based on Lark)
 * backward looking modeling with AR / ECM processes
 * Dulmage - Mendelsohn block decomposition of the model
 * Symbolic computation of the jacobian 
 * Several choices of numerical solvers (based on Scipy, or high-order Newton methods)
 * Time-series management based on Pandas

## Usage

A macro model is defined by a set of static and dynamic equations, which determines the evolution of economic variables (such as GDP, interest rate, etc). The toolbox is able to simulate a trajectory (yearly or quarterly) of a model, based on a sample of time series (a training set). With this training set, the coefficients of the dynamic equations can be estimated, and the residuals of the model computed.      

## Getting started 

 * Clone the repository 
 ~~~ 
 git clone https://github.com/InseeFrLab/Macronometrics.git 
 ~~~

  * Install the package
  
 ~~~
 python setup.py install
 ~~~
 
 * Clone the repository containing an illustrative model
 
 ~~~
 https://github.com/InseeFrLab/Macronometrics-Notebook.git
 ~~~
 
 * Run the Jupyter notebook ```Colibri.ipynb```
 

## To do 

 * Numba just-in-time compilation of the solving functions
 * Estimation of the coefficients of the model (OLS)

## Acknowledgements 

The code for Dulmage - Mendelsohn block decomposition is implemented with courtesy of Bank of Japan research team :

Hirakata, N., K. Kazutoshi, A. Kanafuji, Y. Kido, Y. Kishaba, T. Murakoshi, and T. Shinohara (2019) "The Quarterly Japanese Economic Model (Q-JEM): 2019 version" Bank of Japan Working Paper Series, No. 19-E-7.

Some features of the toolbox are inspired from the Grocer package for Scilab, and implemented with courtesy of Eric Dubois, lead developer of Grocer : http://dubois.ensae.net/grocer.html 

## Credits

Institut National de la Statistique et des Etudes Economiques  
Direction des Etudes et Synthèses Economiques  
Département des Etudes Economiques  
Division des Etudes Macroéconomiques

Benjamin Favetto ([@BFavetto](https://github.com/BFavetto)) - Adrien Lagouge - Matthieu Lequien ([@MLequien](https://github.com/MLequien)) - Olivier Simon




