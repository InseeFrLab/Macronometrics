import setuptools
import os

def strip_comments(l):
    return l.split('#', 1)[0].strip()

def reqs(*f):
    return list(filter(None, [strip_comments(l) for l in open(
        os.path.join(os.getcwd(), *f)).readlines()]))

install_requires = reqs('requirements.txt')

setuptools.setup(
    name = "macronometrics",
    packages = ["macronometrics"],
    version = "0.0.1",
    description = "Toolbox for macroeconometric modeling",
    author = "Benjamin Favetto Adrien Lagouge Olivier Simon",
    author_email = "dg75-g220@insee.fr",
    url = "http://www.insee.fr/",
    download_url = "",
    keywords = ["macroeconomics", "economic modeling", "time series"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    install_requires=install_requires,
    long_description = """\
A toolbox for macroeconometric modeling
---------------------------------------

 * High-level language for model description (parser based on Lark)
 * backward looking modeling with AR / ECM processes
 * Dulmage - Mendelsohn block decomposition of the model
 * Symbolic computation of the jacobian 
 * Several choices of numerical solvers (based on Scipy, or high-order Newton methods)
 * Time-series management based on Pandas
 * Cython / Numba compilation of the solving functions
 * Estimation of the coefficients of the model (OLS)

This version requires Python 3.6 or later.
""",
    python_requires='>=3.6'
)