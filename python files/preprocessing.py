import numpy as np
import pandas as pd


def get_clean_df() -> None:
    """ Reads the train.csv and stores the correctly preprocessed DataFrame in a .csv file."""
    df = pd.read_csv("../data/train.csv", header=0, delimiter=',')
    df = remove_duplicates(df)
    df = remove_nans(df)
    # df = remove_nans_avg(df)  # the other one works better

    week_nums = list(map(get_week_num, list(df.year_week)))
    df.insert(1, 'week_num', week_nums)
    df = df.drop(['date', 'year_week'], axis='columns')

    df.to_csv('../data/preprocessed_data.csv', index = False)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """ Returns pandas.DataFrame without duplicate elements."""
    to_drop = []
    sales = 0.
    inventory = 0.

    for idx in range(len(df)):    
        item = df.iloc[idx]

        if idx+1 < len(df)  and  df.id[idx] == df.id[idx+1]:
            sales += item['sales_units']
            inventory += item['inventory_units']
            to_drop.append(idx)

        else:
            df.loc[idx, 'sales_units'] += sales
            df.loc[idx, 'inventory_units'] += inventory
            sales = 0.
            inventory = 0.

    df = df.drop(to_drop)
    df = df.reset_index(drop=True)

    return df


def remove_nans(df: pd.DataFrame) -> pd.DataFrame:
    """ Returns pandas.DataFrame without nan elements. Gives better results than remove_nans_avg()."""
    for idx in range(1, len(df)):
        item = df.iloc[idx]

        if np.isnan(item['sales_units']):
            df.loc[idx, 'sales_units'] = max(0., df.loc[idx-1,'inventory_units'] - item['inventory_units'])

        if np.isnan(item['inventory_units']):
            df.loc[idx, 'inventory_units'] = max(0., df.loc[idx-1,'inventory_units'] - item['sales_units'])

    return df


def remove_nans_avg(df: pd.DataFrame) -> pd.DataFrame:
    """ Returns pandas.DataFrame without nan elements. Replaces with the average of the previous and the next
        (if they correspond to the same product). Provides worse results than the other method."""
    for idx in range(len(df)):
        item = df.iloc[idx]
        prev = df.iloc[idx-1] if idx > 0 else None
        next = df.iloc[idx+1] if idx < len(df)-1 else None

        if np.isnan(item['sales_units']):
            adjacent_sales = []

            if prev is not None  and  prev.product_number == item.product_number  and  not np.isnan(prev.sales_units):
                adjacent_sales.append(prev.sales_units)
            if next is not None  and  next.product_number == item.product_number  and  not np.isnan(next.sales_units):
                adjacent_sales.append(next.sales_units)

            df.loc[idx, 'sales_units'] = np.average(adjacent_sales)

        if np.isnan(item['inventory_units']):
            adjacent_inv = []

            if prev is not None  and  prev.product_number == item.product_number  and  not np.isnan(prev.inventory_units):
                adjacent_inv.append(prev.inventory_units)

            if next is not None  and  next.product_number == item.product_number  and  not np.isnan(next.inventory_units):
                adjacent_inv.append(next.inventory_units)

            df.loc[idx, 'inventory_units'] = np.average(adjacent_inv)

    return df


def get_week_num(n: int) -> int:
    """ Returs the number of weeks since the start of 2019 corresponding to the week n = YYYYWW."""
    y = n // 100 - 2019
    w = n % 100
    return w + 52*y


def processed_dataframe() -> pd.DataFrame:
    """ Returns the preprocessed DataFrame. The function get_clean_df() MUST have been called previously."""
    return pd.read_csv("../data/preprocessed_data.csv", header=0, delimiter=',')


def create_characteristics_and_inventory_matrix(df: pd.DataFrame) -> tuple[ pd.DataFrame, list[list[int]], dict[int,int] ]:
    """ Returns a DataFrame of the characteristics of each product, a Matrix of the different inventories for each product,
        and a dictionary relating each product with its index in both the DataFrame and the Matrix."""
    inv_matrix = []
    product_dict = {}
    prev = 0
    i = 0
    
    characteristics = pd.DataFrame( columns=list(df.columns[2:-2]) + ['avg_sales', 'avg_inventory'] )

    for idx in range(len(df)):
        if idx+1 < len(df) and df.product_number[idx] == df.product_number[idx+1]:  # this product's inventory will be later computed
            continue

        sales = np.array(df.sales_units[prev:idx+1])
        inventory = np.array(df.inventory_units[prev:idx+1])
        inv_matrix.append(inventory)

        item = df.iloc[idx]
        characteristics.loc[i] = [item[c] for c in characteristics.columns[:-2]] + [np.average(sales), np.average(inventory)]
        
        product_dict[df.product_number[idx]] = i
        i += 1
        prev = idx+1

    return characteristics, inv_matrix, product_dict


if __name__ == '__main__':
    get_clean_df()
    df = processed_dataframe()
    
    # characteristics, inv_matrix, product_dict = create_characteristics_and_inventory_matrix(df)    
    # print(*inv_matrix[:1], sep = '\n', end = '\n\n')
    # print(characteristics.head(), sep = '\n', end = '\n\n')
    # print(product_dict, end = '\n\n')
    # characteristics.to_csv('../data\ analysis/characteristics.csv', index = False)

