# AlNostreRitme

The goal of this challenge is to forecast the inventory of some products during 13 weeks (indirectly, this also means predicting sales), based on previous sales and inventory data and also on the properties of the product.

Instead of taking an ML approach to this, we have decided to adapt and use a concept from Signal Processing: the adaptive filter. This type of filter is used for non-stationary signal prediction. The filter uses a finite number of past observations to predict the next observation, and then uses the error to correct the filter coefficients, with the objective of minimizing the MSE (which is equivalent to our goal of minimizing the RMSE). Our iterative algorithm is the following: $$\textbf{h}^{n+1} = \textbf{h}^{n} + \mu\textbf{h}[n]e[n]$$
Where $\textbf{h}$ is the filter (a vector with L coefficients), $\textbf{x}[n]$ is the vector $[x[n-L+1], ..., x[n]]$, $e[n]$ is the prediction error for observation n, and $\mu$ is the step size (which affects how fast the filter converges as well as its misadjustment).

This approach is similar to the ARIMA methodology in the sense that we use a linear combination of past observations to predict a future observation, and then recursively use this prediction to obtain the next prediction. However, we decided to use the concept of the adaptive filter becasue it provided certain advantages: the coefficients of our filter adapt, which can help with predicitons, and, additionally, ARIMA requires that we validate the models before making predictions, which is very time consuming, especially if we have to create models for the different products.

There are 2 parameters that we need to determine for our filter: its length $L$, and the value of $\mu$. The approach that we took consists of trying different reasonable values of L and $\mu$ (if L is too large the filter won't be able to adapt to changes quickly, and if $\mu$ is too large or too small the filter won't converge).

## Preprocessing

Since our data has some inconsistencies, we first need to preprocess it, although we won't do it as rigorously as we would if we applied ML, because our methodology does not require it.

### preprocessing.py
This file includes all the functions that allow us to obtain a clean dataframe without null values or repeated ids in the same week.

It's based on 2 functions. The first one, remove_duplicates(), deals with duplicated ids. Sometimes, there are two rows with the same week and product number, associated to different resellers. Since we only need to predict the inventory based on the product and the week, and not the reseller, we will group these rows into one. The function iterates over all the dataframe, dropping all the rows with a repeated product and week, and increasing the total sells and inventory units for the one instance that we keep (essentially a GroupBy).

The second function, remove_nans(), also iterates over our dataframe and whenever it finds a null value, either on the sales_units or on the inventory_units, it tries to approximate its value using previous and following data (or just the previous).

Another very important part of this file is the last function, which returns a dataframe, a matrix, and a dictionary. The dataframe returned has a unique entry for each product, and it stores all of the product's properties, such as its product number, reporter id, product category, specs, display size, segment, and the average of both sales and inventory units. The matrix that is returned contains, for each product, a list of all the inventory units that were on the original dataframe. Finally, the dictionary contains the product number for each product as its keys, and as values, the index that this product was assigned on both the characteristics dataframe and the matrix. These data structures are later used for prediction.

### plotting.py
This file contains various plotting functions that allowed us to have a better understanding on how the given data was structured.
The plotting functions include some plotting of the preprocessed data, some plotting of the distribution of the different characteristics of the products stated before, and some plots of the data sorted by which different characteristic they fulfilled.


## Original Approach

Our original approach is what we mentioned in the introduction, and its implementation can be found in original.py and original.ipynb. In this case, we have separate filters for each product_number. We use the adaptive filter to compute predictions, and select L and $\mu$ based on how well the filter can predict the observations that we have. However, we give more importance to the last predictions being the best. We chose this because we want our filter to adjust well with the final observations that we have available, because those are the ones that will be most correlated with the future observations that we want to predict. We also tried different metrics, but this proved to be the one that gave us the best RMSE results.

## Modified Approach

In this approach, we evaluate the different filters using a different criteria, and its implementation can be found in new_criteria.py and new_criteria.ipynb. We also select one filter (with different parameters) for each type of product, but now we choose the best filter by evaluating its ability to predict the last observation only. Additionally, the filter is sensitive to initialization conditions, and in the previous case we initialized it with a vector of zeros, so in this case we also try some other initial vectors. This new approach provides better results.

## Clustering

Our previous approaches treat each product separately, instead of looking at the possible correlations due to shared attributes (category, specs, etc.). We tried to incorporate these variables into our analysis, which can be seen in the data analysis folder. First, we applied Multidimensional Scaling to look at the relationships between attributes and individuals. Next, we used the results from Multidimensional Scaling to apply K-means to cluster our data into bigger categories.

Our idea was to simplify our problem by, instead of creating one filter for each product, creating one filter per cluster, and assuming that the products that belong to the same cluster follow similar trends. We have not had time to develop a fully functional implementation of this idea, but an initial proposal can be found in clustering.ipynb.
