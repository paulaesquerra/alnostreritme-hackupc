import numpy as np
import pandas as pd

from preprocessing import processed_dataframe, create_characteristics_and_inventory_matrix

### Functions Predict ###
def LMSalgorithm(x, K, L, init): 
    """This method computes analogues results than the one with the same name in 'original.py' but with two modifications. First, 'init' parameter changes the initial
    filter coeficcients. And, secondly, the error metric used is only the RMSE of the last prediction."""
    L = int(L)
    total_error = 0
    iterations = 0
    Nr = x@x
    if Nr == 0:
        Nr = Nr +0.0000000001
    mu = 2/Nr*K
    i = 0
    if init == 0:
        h = np.zeros(L)
    elif init == 1:
        h = 0.5*np.ones(L)
    else:
        h = -0.5*np.ones(L)
    x_n = x[i:i+L]
    while (i+L < len(x)):
        d = x[i+L]
        x_n = x[i:i+L]
        y = x_n@h
        if y%2 < 0.5:
            y = y+1
        y = int(y)
        #y = round(y)
        e = d-y
        if len(x)-i-L == 1:
            total_error = total_error + e*e
            iterations = iterations+1
        h = h+mu*x_n*e
        i = i+1
    #pred = round(x_n@h)
    return np.sqrt(total_error/iterations), x_n@h


def LMSalgorithm_adaptativeMU(x, K, L, init):
    """This method is the equivalent to the one with same name in 'original.py' applying the modifications mentioned in the previous function. """
    L = int(L)
    total_error = 0
    iterations = 0
    i = 0
    if init == 0:
        h = np.zeros(L)
    elif init == 1:
        h = 0.5*np.ones(L)
    else:
        h = -0.5*np.ones(L)
    x_n = x[i:i+L]
    while (i+L < len(x)):
        d = x[i+L]
        x_n = x[i:i+L]
        y = x_n@h
        if y%2 < 0.5:
            y = y+1
        y = int(y)
        # y = round(y)
        e = d-y
        if len(x)-i-L == 1:
            total_error = total_error + e*e
            iterations = iterations+1
        Nr = x_n@x_n
        if Nr == 0:
            Nr = Nr+0.0000000001
        mu = 2/Nr*K
        h = h+mu*x_n*e
        i = i+1
    #pred = round(x_n@h)
    return np.sqrt(total_error/iterations), x_n@h


def IPA_prediction(x):
    """As in 'original.py' this function aims to find the best model to predict the following samples of 'x'. However, in this case there is also a list of values 
    'initialization' that indicate different initial filters over wich we also iterate."""
    possible_K = np.array([1/2, 1/3, 1/5, 1/10, 1/50, 1/100])
    possible_L = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    initialization = np.array([0,1,2])

    min_error = 100000000
    for k in range(len(possible_K)):
        for l in range(len(possible_L)):
            for i in range(len(initialization)):
                [rmse, pred] = LMSalgorithm(x, possible_K[k], possible_L[l], initialization[i]) 
                if rmse < min_error:
                    min_error = rmse
                    best_L = possible_L[l]
                    best_K = possible_K[k]
                    best_algorithm = LMSalgorithm
                    best_init = initialization[i]
                [rmse, pred] = LMSalgorithm_adaptativeMU(x, possible_K[k], possible_L[l], initialization[i])
                if rmse < min_error:
                    min_error = rmse
                    best_L = possible_L[l]
                    best_K = possible_K[k]
                    best_algorithm = LMSalgorithm_adaptativeMU
                    best_init = initialization[i]
    return best_L, best_K, best_algorithm , best_init


def get_results_matrix(inv_matrix: list) -> list:
    """ Returns a matrix with 13 columns, one for each week, and 100 rows, one for each product."""

    res_matrix = [[] for _ in range(13)]
    for i in range(len(inv_matrix)):
        x = inv_matrix[i][:]
        for j in range(13):
            best = IPA_prediction(x)
            [rmse , pred]=best[2](x, best[1], best[0], best[3])
            res_matrix[j].append(pred)
            x = np.block([x, pred])

    return res_matrix


def save_results_csv(res_matrix: list, product_dict: dict) -> None:
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
    df_final.to_csv('../submissions/submission7.csv', index = False)


if __name__ == '__main__':
    IPA_data = processed_dataframe() # you MUST have executed preprocessing.py before
    _, inv_matrix, product_dict = create_characteristics_and_inventory_matrix(IPA_data)

    res_matrix = get_results_matrix(inv_matrix)

    save_results_csv(res_matrix, product_dict)






