# -*- coding: utf-8 -*-
import numpy as np

#Création de la matrice data(elle recense les alternatives et les critères)
def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)

# Étape de pondération
# il faut pouvoir justifier les poids qui ont été donnés
def weigh(data, weights):
    return data * weights

# Étape de création de matrices de flux
def positive_flow(data):
    n = data.shape[0]
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p_matrix[i][j] = np.sum(np.maximum(data[j] - data[i], 0))
    return p_matrix

def negative_flow(data):
    n = data.shape[0]
    n_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            n_matrix[i][j] = np.sum(np.maximum(data[i] - data[j], 0))
    return n_matrix

# Étape de calcul des indices de préférence
def preference_index(p_matrix, n_matrix):
    n = p_matrix.shape[0]
    s_plus = np.sum(p_matrix, axis=1)
    s_minus = np.sum(n_matrix, axis=1)
    index = np.zeros(n)
    for i in range(n):
        index[i] = s_minus[i] / (s_minus[i] + s_plus[i])
    return index


# Classement final
def promethee(data, weights):
    n_data = normalize(data)
    w_data = weigh(n_data, weights)
    p_matrix = positive_flow(w_data)
    n_matrix = negative_flow(w_data)
    pref_index = preference_index(p_matrix, n_matrix)
    ranking = np.argsort(pref_index)[::-1]
    return ranking
