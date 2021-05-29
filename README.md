# Rossmann-Sales-Prediction
Capstone project 2 -  Sales Prediction : Predicting sales of a major store chain Rossmann

Rossmann Sales prediction project is made by myself and my teammate - shafil. When we look the problem statement and the dataset, firstly we discuss about problem statement and the approach to do this project. I took the part for doing EDA and do feature engineering. For model formulation and training model with various algorithms to get best model, this part is done by my mate Shafil.

Now, after looking datasets i can found that there were two datasets are given. the first dataset is of Rossmann Stores Data.csv - which contains the historical data including Sales and the second dataset contain store.csv - which has supplemental information about the stores.

The Shape of the Rossmann Stores Data.csv is (1017209,9) and shape of the store.csv is (1115,10). On looking for Nan values, I found that Store dataset has lots of Nan values. So I try to replacing this Nan values with suitable values and makes datasets good.

After that I tried to merge the both datasets into single, so Store column is common to both the datasets, I tried to join the datasets by inner join. Now I obtained final datasets which have now shape (1017209,18). After getting this dataset I do EDA on it and got very useful results which was shown in this notebook.
