from pandas import offsets, date_range, to_datetime, DataFrame
import numpy as np
from math import floor

lag_trim = offsets.MonthBegin(3)

def extrapol_affine(data_init, nom_serie, multi, addi, date_start, date_end):
    """
    Prolonge la base data_init par extrapolation des séries contenues dans nom_serie 
    en les multipliant par multi et en additionnant addi
    entre les dates date_start et date_end (définies comme chaînes de caractères au format YYYYQ)
    """
    data_extrapol = data_init.copy()
    iter_dates = date_range(start=to_datetime(
        date_start), end=to_datetime(date_end), freq="QS")

    for item in iter_dates:
        data_extrapol.loc[item, nom_serie] = data_extrapol.loc[item -
                                                               lag_trim, nom_serie]*multi + addi

    return data_extrapol


def extrapol_prolonge_taux_moyen(data_init, nom_serie, date_t1, date_t2, date_start, date_end):
    """
    Prolonge la base data_init par extrapolation en taux de croissance 
    des séries contenues dans nom_serie entre date_start jusqu'à l'horizon voulu date_end, au taux moyen observé
    entre t1+1 et t2 
    Dates au format YYYYQ
    """
    data_extrapol = data_init.copy()
    # Attention au décalage
    serie_tx = (data_extrapol[nom_serie].shift(-1) /
                data_extrapol[nom_serie] - 1).copy()

    # Calcul du taux moyen sur la période considérée
    tx_moyen = serie_tx[(to_datetime(date_t1) + lag_trim):to_datetime(date_t2)].mean()

    # Prolongement de la série
    iter_dates = date_range(start=to_datetime(
        date_start), end=to_datetime(date_end), freq="QS")
    temp = data_extrapol.loc[to_datetime(
        date_start)-lag_trim, nom_serie].copy()

    for item in iter_dates:
        temp *= (1+tx_moyen)
        data_extrapol.loc[item, nom_serie] = temp

    return data_extrapol


def extrapol_prolonge_constant(data_init, nom_serie, date_start, date_end, value='last'):
    """
    Prolonge la série de date_start jusqu'à l'horizon voulu date_end à une valeur constante
    égale par défaut à la dernière valeur observée ou à la valeur "value" si spécifié autrement
    Dates au format YYYYQ
    """
    data_extrapol = data_init.copy()

    iter_dates = date_range(start=to_datetime(
        date_start), end=to_datetime(date_end), freq="QS")
    last_value = data_extrapol.loc[to_datetime(
        date_start)-lag_trim, nom_serie]

    for item in iter_dates:
        if value == 'last':
            data_extrapol.loc[item, nom_serie] = last_value
        else:
            data_extrapol.loc[item, nom_serie] = value

    return data_extrapol


def extrapol_duplique(data_init, nom_serie, date_start, date_end, vect):
    """
    Prolonge la série de date_start jusqu'à l'horizon voulu date_end en dupliquant le vecteur vect
    Dates au format YYYYQ
    """
    data_extrapol = data_init.copy()

    rpt = len(data_extrapol[to_datetime(date_start):to_datetime(date_end)+lag_trim])
    data_extrapol.loc[to_datetime(date_start):to_datetime(
        date_end), nom_serie] = np.tile(vect, floor(rpt/4))

    return data_extrapol


def extrapolate_series(data_init, liste_series):
    """
    Prolonge toutes les séries définies dans liste_series selon le mode d'extrapolation spécifié
    Un élément de liste_series doit être écrit sous la forme :
    - [nom_serie,'constant',[date_start,date_end],value] -> prolongation constante à la dernière valeur observée ou à la valeur value
    - [nom_serie, 'taux de croissance',[date_start,date_end],taux] -> prolongation selon un taux de croissance constant indiqué par taux (en %)
    - [nom_serie, 'taux de croissance moyen',[date_start,date_end],[date_t1,date_t2]]->prolongation selon un taux de croissance constant moyen calculé entre date_t1 et date_t2
    - [nom_serie, 'affine',[date_start,date_end],multi,addi]->prolongation en multipliant par multi et en additionnat addi
    - [nom_serie, 'dummy_trim',[date_start,date_end],vect]->prolongation en dupliquant le vecteur vect
    """
    data_extrapol = data_init.copy()

    for item in liste_series:
        if item[1] == 'constant':
            data_extrapol = extrapol_prolonge_constant(
                data_init=data_extrapol, nom_serie=item[0], date_start=item[2][0], date_end=item[2][1], value=item[3])
        elif item[1] == 'taux de croissance':
            data_extrapol = extrapol_affine(
                data_init=data_extrapol, nom_serie=item[0], multi=1+0.01*item[3], addi=0, date_start=item[2][0], date_end=item[2][1])
        elif item[1] == 'taux de croissance moyen':
            data_extrapol = extrapol_prolonge_taux_moyen(
                data_init=data_extrapol, nom_serie=item[0], date_t1=item[3][0], date_t2=item[3][1], date_start=item[2][0], date_end=item[2][1])
        elif item[1] == 'affine':
            data_extrapol = extrapol_affine(
                data_init=data_extrapol, nom_serie=item[0], multi=item[3], addi=item[4], date_start=item[2][0], date_end=item[2][1])
        else:
            data_extrapol = extrapol_duplique(
                data_init=data_extrapol, nom_serie=item[0], date_start=item[2][0], date_end=item[2][1], vect=item[3])

    return data_extrapol


def compare_series(type_comparaison, db_init, db_fin, liste_var, date_init, date_fin):
    """
    Permet de comparer les valeurs des séries dont les noms sont contenues dans liste_var
    entre les dataframes db_init et db_fin soit :
    - en écart absolu si type_comparaison='niveau'
    - en écart relatif (db_fin/db_init-1)*100 si type_comparaison='relatif'
    Suppose que les dataframe db_init et db_fin aient les mêmes dimensions et la même indexation temporelle
    """
    results_trim = DataFrame()

    for serie in liste_var:
        if type_comparaison == 'absolu':
            results_trim.loc[to_datetime(date_init):to_datetime(date_fin), serie] = db_fin.loc[to_datetime(
                date_init):to_datetime(date_fin), serie]-db_init.loc[to_datetime(date_init):to_datetime(date_fin), serie]
        else:
            results_trim.loc[to_datetime(date_init):to_datetime(date_fin), serie] = 100*(db_fin.loc[to_datetime(
                date_init):to_datetime(date_fin), serie]/db_init.loc[to_datetime(date_init):to_datetime(date_fin), serie]-1)

    results_trim = results_trim.dropna()

    results_ann = results_trim.groupby(results_trim.index.year).mean()

    return (results_trim, results_ann)
