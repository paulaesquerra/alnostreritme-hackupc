import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest

from preprocessing import processed_dataframe, create_characteristics_and_inventory_matrix


### Global plotting ###

def plot_sales_and_inv(df: pd.DataFrame) -> None:
    """ Prints the sales and inventory lists correspondig to each product."""
    prev = 0
    for idx, (num1, num2) in enumerate(zip_longest(df.product_number, df.product_number[1:])):
        if num1 != num2:
            plt.plot(df['week_num'][prev:idx+1], df['sales_units'][prev:idx+1], label = 'sales', c = 'green', alpha = 0.3)
            plt.plot(df['week_num'][prev:idx+1], df['inventory_units'][prev:idx+1], label='inventory', c = 'orange', alpha = 0.3)
            prev = idx+1
    plt.title('Sales units and inventory units')
    plt.show()
    
    df.sales_units.plot(kind = 'hist')
    df.inventory_units.plot(kind='hist', alpha = 0.5, bins = 26)
    plt.title('Sales units and inventory units')
    plt.show()



def plot_segments(df: pd.DataFrame) -> None:
    """ Plots all the inventory units for each product, color-coded with its segment."""
    prev = 0
    for idx, (num1, num2) in enumerate(zip_longest(df.product_number, df.product_number[1:])):
        if num1 != num2:
            plt.plot(df['week_num'][prev:idx+1], df['inventory_units'][prev:idx+1],  c = get_color(df.loc[idx, 'segment']), alpha = 0.3)
            prev = idx+1

    for seg in ['Premium', 'Core', 'Gaming']:
        plt.plot([], label = seg, c = get_color(seg))
    plt.title('Segments')
    plt.legend()
    plt.show()


def get_color(segment: str) -> str:
    if segment == 'Premium': return 'orangered'
    if segment == 'Core': return 'green'
    if segment == 'Gaming': return 'blue'


### Plotting by product characteristics ###

def plot_prod_inv(prod_num: int, inv_matrix: list, prod_dict: dict) -> None:
    """ Plots the product inventory corresponding to the prod_num."""
    a = inv_matrix[prod_dict[prod_num]]
    plt.plot(np.arange(226-len(a), 226), a)
    plt.xlim(10, 230)


def reporterhq_id_comparison(df: pd.DataFrame, characteristics: pd.DataFrame) -> None:
    """ Comparison of the reporterhhq_id in both DataFrames. We can see that in the characteristics DataFrame, there is less variety
        and that higher values are overrepresented."""
    print(characteristics.reporterhq_id.describe())
    plt.subplot(1, 2, 1, title = 'Reporter id at the original df')
    df.groupby(['reporterhq_id']).count().product_number.plot(kind ='bar')

    plt.subplot(1, 2, 2, title = 'Reporter id at characteristics')
    characteristics.groupby(['reporterhq_id']).count().product_number.plot(kind ='bar')
    plt.show()


def describe_and_plot_prod_category(df: pd.DataFrame, characteristics: pd.DataFrame, inv_matrix: list, product_dict: dict) -> None:
    """ Describes the important statistical parameters of the prod_category characteristic, and plots their distribution.
        Plots the inventories of the products sorted out by their product category"""
    print(characteristics.prod_category.describe())

    characteristics.groupby(['prod_category']).count().product_number.plot(kind ='bar', label = 'product category')
    plt.axhline(characteristics.groupby(['prod_category']).count().product_number.mean(), color='r', linestyle='-', label = 'mean')
    plt.legend()
    plt.title('Product category')
    plt.show()

    for prods in characteristics.groupby(['prod_category']):
        for p in prods[1].iterrows():
            plt.title('Product category = '+str(p[1].prod_category))
            plot_prod_inv(p[1].product_number, inv_matrix, product_dict)
        plt.show()


def describe_and_plot_specs(df: pd.DataFrame, characteristics: pd.DataFrame) -> None:
    """ Desciribes the important statistical parameters of the specs characteristic, and prints a histogram of its distriburion."""
    print(characteristics.specs.describe())

    characteristics.specs.plot(kind = 'hist' , label = 'specs')
    plt.axvline(characteristics.specs.mean(), color='r', label = 'mean')
    plt.legend()
    plt.title('Specs')
    plt.show()


def describe_and_plot_display_size(df: pd.DataFrame, characteristics: pd.DataFrame, inv_matrix: list, product_dict: dict) -> None:
    """ Desciribes the important statistical parameters of the display_size characteristic, and plots their distriburion.
        Plots the inventories of the products sorted out by their display size."""
    print(characteristics.display_size.describe())

    characteristics.groupby(['display_size']).count().product_number.plot(kind ='bar', label = 'display size')
    plt.axhline(characteristics.display_size.mean(), color='r', linestyle='-', label = 'mean')
    plt.legend()
    plt.title('Display size')
    plt.show()

    for prods in characteristics.groupby(['display_size']):
        for p in prods[1].iterrows():
            plt.title('Display size = '+str(p[1].display_size))
            plot_prod_inv(p[1].product_number, inv_matrix, product_dict)
        plt.show()


def describe_and_plot_segment(df: pd.DataFrame, characteristics: pd.DataFrame, inv_matrix: list, product_dict: dict) -> None:
    """ Desciribes the important statistical parameters of the segment characteristic, and plots their distriburion.
        Plots the inventories of the products sorted out by their segment."""
    print(characteristics.segment.describe())

    characteristics.groupby(['segment']).count().product_number.plot(kind ='bar', label = 'display size')
    plt.axhline(characteristics.groupby(['segment']).count().product_number.mean(), color='r', linestyle='-', label = 'mean')
    plt.legend()
    plt.title('Segment')
    plt.show()

    for prods in characteristics.groupby(['segment']):
        for p in prods[1].iterrows():
            plt.title('Segment = '+str(p[1].segment))
            plot_prod_inv(p[1].product_number, inv_matrix, product_dict)
        plt.show()


def plot_avg_sales_and_avg_inv(characteristics: pd.DataFrame) -> None:
    """ Plots the average sales and the average inventory for each product in the characteristics DataFrame."""
    characteristics.avg_inventory.plot(label = 'avg inventory')
    characteristics.avg_sales.plot(label = 'avg sales')
    plt.title('Average sales and average inventory units')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    df = processed_dataframe()
    plot_sales_and_inv(df)
    plot_segments(df)

    characteristics, inv_matrix, product_dict = create_characteristics_and_inventory_matrix(df)
    plot_prod_inv(6909, inv_matrix, product_dict)
    plot_prod_inv(7896, inv_matrix, product_dict)
    plot_prod_inv(233919, inv_matrix, product_dict)

    reporterhq_id_comparison(df, characteristics)

    describe_and_plot_prod_category(df, characteristics, inv_matrix, product_dict)
    
    describe_and_plot_specs(df, characteristics)

    describe_and_plot_display_size(df, characteristics, inv_matrix, product_dict)

    describe_and_plot_segment(df, characteristics, inv_matrix, product_dict)
    
    plot_avg_sales_and_avg_inv(characteristics)
