import numpy as np
import pandas as pd

from preprocessing import processed_dataframe, create_characteristics_and_inventory_matrix


### Functions predict ###
def LMSalgorithm(x, K, L):
    """ LMS algorithm:  Given a realization 'x' of a stochastic process, this algorithm computes an adaptive filter 'h' of length 'L', which is used to predict
    the following sample of the realization. 'K' is used as a hyperparameter related to 'mu'. In this case we compute 'mu' only once, using the full realizaton.
    The function also returns an error metric which is RMSE wheighted by the square inverse of the distance to the new sample we want to predict. """
    L = int(L)
    total_error = 0
    iterations = 0
    #print(L, K)
    #print(total_error, iterations)
    Nr = x@x
    # To avoid division by 0:
    if Nr == 0:
        Nr = Nr +0.0000000001
    mu = 2/Nr*K
    i = 0
    h = np.zeros(L)
    x_n = x[i:i+L]
    while (i+L < len(x)):
        d = x[i+L]
        x_n = x[i:i+L]
        y = x_n@h
        # Rounding prediction value:
        if y%2 < 0.5:
            y = y+1
        y = int(y)
        e = d-y
        total_error = total_error + e*e*1/(len(x)-i-L)**2
        iterations = iterations+1
        h = h+mu*x_n*e
        i = i+1
    return np.sqrt(total_error/iterations), x_n@h


def LMSalgorithm_adaptativeMU(x, K, L):
    """ LMS algorithm with adaptative mu: Given a realization 'x' of a stochastic process, this algorithm computes an adaptive filter 'h' of length 'L', which is used
    to predict the following sample of the realization. 'K' is used as a hyperparameter related to 'mu'. In this case we compute 'mu' for each iteration, using only
    a moving window of length 'L' of the realization. The function also returns an error metric which is RMSE wheighted by the square inverse of the distance to the new 
    sample we want to predict. """
    L = int(L)
    total_error = 0
    iterations = 0
    i = 0
    h = np.zeros(L)
    x_n = x[i:i+L]
    while (i+L < len(x)):
        d = x[i+L]
        x_n = x[i:i+L]
        y = x_n@h
        # Rounding prediction value:
        if y%2 < 0.5:
            y = y + 1
        y = int(y)
        e = d-y
        total_error = total_error + e*e*1/(len(x)-i-L)**2
        iterations = iterations+1
        Nr = x_n@x_n
        # To avoid division by zero:
        if Nr == 0:
            Nr = Nr+0.0000000001
        mu = 2/Nr*K
        h = h+mu*x_n*e
        i = i+1
    return np.sqrt(total_error/iterations), x_n@h


def IPA_prediction(x):
    """ IPA prediction: This method is used to find the best model to predict the following sample of 'x' which is the ones that minimizes the RMSE from the previous
    functions. Each model is determined by the algorithm used (with constant or adaptive 'mu'), length 'possible_L' of the filter and the hyperparameter in 'possile_K'.
    It iterates using the values we have selected."""
    possible_K = np.array([1/2, 1/3, 1/5, 1/10, 1/50, 1/100])
    possible_L = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 

    min_error = 100000000
    for k in range(len(possible_K)):
        for l in range(len(possible_L)):
            [rmse, pred] = LMSalgorithm(x, possible_K[k], possible_L[l]) 
            if rmse < min_error:
                min_error = rmse
                best_L = possible_L[l]
                best_K = possible_K[k]
                best_algorithm = LMSalgorithm
            [rmse, pred] = LMSalgorithm_adaptativeMU(x, possible_K[k], possible_L[l])
            if rmse < min_error:
                min_error = rmse
                best_L = possible_L[l]
                best_K = possible_K[k]
                best_algorithm = LMSalgorithm_adaptativeMU
    return best_L, best_K, best_algorithm


### IPA with neat dataset ###

def get_results_matrix(inv_matrix):
    """ Returns a matrix with 13 columns, one for each week,
    and 100 rows, one for each product."""
    res_matrix = [[] for _ in range(13)]
    for i in range(len(inv_matrix)):
        x = inv_matrix[i][:]
        for j in range(13):
            best = IPA_prediction(x)
            [rmse , pred]=best[2](x, best[1], best[0])
            res_matrix[j].append(pred)
            x = np.block([x, pred])
    
    return res_matrix


def save_results_csv(res_matrix, product_dict):
    """ Saves the results, that can be obtained from the result matrix and the
        dictionary of product indexes."""
    product_dict = dict(map(lambda x: (str(x[0]), x[1]), product_dict.items()))
    new_prod_dict = dict(sorted(product_dict.items()))

    week = 202319
    final_dict = {}
    for i in range(13):
        for item in new_prod_dict:
            final_dict[str(week+i)+"-"+item] = res_matrix[i][new_prod_dict[item]]

    final_id = []
    final_items = []
    for item in final_dict:
        final_id.append(item)
        final_items.append(final_dict[item])

    df_final = pd.DataFrame({'id': final_id, 'inventory_units': final_items})
    df_final.to_csv('../submissions/submission4.csv', index = False)


if __name__ == '__main__':
    IPA_data = processed_dataframe() # you MUST have executed preprocessing.py before
    _, inv_matrix, product_dict = create_characteristics_and_inventory_matrix(IPA_data)

    res_matrix = get_results_matrix(inv_matrix)

    save_results_csv(res_matrix, product_dict)
    
    
