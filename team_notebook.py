# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rossmann_df = pd.read_csv('Rossmann Stores Data.csv')
store_df = pd.read_csv('store.csv')

rossmann_df.head()

rossmann_df.shape

"""We see here that the rossmann dataset consists of 1017209 values and 9 features"""

rossmann_df.info()

store_df.head()

store_df.shape

"""the store dataset consists of 1115 values and 10 features

## Looking for Nan values in Store csv file
"""

store_df.isnull().sum()

"""There are many Nan values in columns - **'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear, Promointerval', 'Promo2sinceWeek' and 'Promo2sinceYear'**. Also **CompetitionDistance** has only 3 null values. we have to clean those data. Let's start checking....

### 1. CompetitionDistance
"""

store_df[pd.isnull(store_df.CompetitionDistance)]

"""So, we can fill these three values with many ways such as 0 or mean or mode or median. We decided to fill with it Median."""

## code for replacing Nan values in CompetitionDistance with mode.
store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].median(), inplace = True)

"""### 2. 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear, Promointerval', 'Promo2sinceWeek' and 'Promo2sinceYear'

There are not much information provided to these data. Also we observe from dataset that where the **Promo2** has value equals to zero there are Nan values for these columns. That means the store which do not wat promotion they have null values in promointerval , promo2sinceweek and so on.So for this purpose the best way to fill those features is to assign value equals to zero.
"""

## code for replacing Nan values with 0.

store_new = store_df.copy()

## Replacing Nan values with 0 in CompetitionOpenSinceMonth
store_new['CompetitionOpenSinceMonth'] = store_new['CompetitionOpenSinceMonth'].fillna(0)

## Replacing Nan values with 0 in CompetitionOpenSinceYear
store_new['CompetitionOpenSinceYear'] = store_new['CompetitionOpenSinceYear'].fillna(0)

## Replacing Nan values with 0 in Promo2SinceWeek
store_new['Promo2SinceWeek'] = store_new['Promo2SinceWeek'].fillna(0)

## Replacing Nan values with 0 in Promo2SinceYear
store_new['Promo2SinceYear'] = store_new['Promo2SinceYear'].fillna(0)

## Replacing Nan values with 0 in PromoInterval
store_new['PromoInterval'] = store_new['PromoInterval'].fillna(0)

## Now checking Nan values
store_new.isna().sum()

"""## Merge the Rossmann_df and Store_df csv by column 'Store' as in both csv Store column is common."""

final1 = pd.merge(rossmann_df, store_new, on='Store', how='left')
final1.head()

final1.shape

"""## Changing different dtypes to int type."""

# code for changing StateHoliday dtype from object to int.
final1.loc[final1['StateHoliday'] == '0', 'StateHoliday'] = 0
final1.loc[final1['StateHoliday'] == 'a', 'StateHoliday'] = 1
final1.loc[final1['StateHoliday'] == 'b', 'StateHoliday'] = 2
final1.loc[final1['StateHoliday'] == 'c', 'StateHoliday'] = 3
final1['StateHoliday'] = final1['StateHoliday'].astype(int, copy=False)

print('levels :', final1['StateHoliday'].unique(), '; data type :', final1['StateHoliday'].dtype)

# code for changing Assortment dtype from object to int.
final1.loc[final1['Assortment'] == 'a', 'Assortment'] = 0
final1.loc[final1['Assortment'] == 'b', 'Assortment'] = 1
final1.loc[final1['Assortment'] == 'c', 'Assortment'] = 2
final1['Assortment'] = final1['Assortment'].astype(int, copy=False)

print('levels :', final1['Assortment'].unique(), '; data type :', final1['Assortment'].dtype)

# code for changing StoreType dtype from object to int.
final1.loc[final1['StoreType'] == 'a', 'StoreType'] = 0
final1.loc[final1['StoreType'] == 'b', 'StoreType'] = 1
final1.loc[final1['StoreType'] == 'c', 'StoreType'] = 2
final1.loc[final1['StoreType'] == 'd', 'StoreType'] = 3
final1['StoreType'] = final1['StoreType'].astype(int, copy=False)

print('levels :', final1['StoreType'].unique(), '; data type :', final1['StoreType'].dtype)

# code for changing format of date from object to datetime
final1['Date'] = pd.to_datetime(final1['Date'], format= '%Y-%m-%d')

final1['CompetitionOpenSinceYear']= final1['CompetitionOpenSinceYear'].astype(int)
final1['Promo2SinceYear']= final1['Promo2SinceYear'].astype(int)

final1['CompetitionOpenSinceMonth'] = pd.DatetimeIndex(final1['Date']).month

final1['CompetitionDistance']= final1['CompetitionDistance'].astype(int)
final1['Promo2SinceWeek']= final1['Promo2SinceWeek'].astype(int)

"""## checking dtypes of columns"""

final1.dtypes

"""# Exploratory Data Analysis"""

final1.head()

final1.describe().apply(lambda s: s.apply('{0:.2f}'.format))

final1.tail()

final1.info

"""### Sales"""

plt.figure(figsize=(15,6))
sns.pointplot(x= 'CompetitionOpenSinceYear', y= 'Sales', data=final1)
plt.title('Plot between Sales and Competition Open Since year')

"""From the Plot we can tell that Sales are high during the year 1900, as there are very few store were operated of Rossmann so there is less competition and sales are high. But as year pass on number of stores increased that means competition also increased and this leads to decline in the sales."""

plt.figure(figsize=(15,6))
sns.pointplot(x= 'Promo2SinceYear', y= 'Sales', data=final1)
plt.title('Plot between Sales and Promo2 Since year')

"""Plot between Sales and promo2 since year shows that effect of sales of stores which continue their promotion. this data is available from yaer 2009 to 2015. Promo2 has very good effect on sales but in year 2013 sales be minimum and also in year 2012 and 2015 sales are very low."""

plt.figure(figsize=(15,6))
sns.pointplot(x= 'CompetitionOpenSinceMonth', y= 'Sales', data=final1)
plt.title('Plot between Sales and Competition Open Since Month')

"""Plot between Competition open since month and Sales explains the sales data in each month of a year. This data shows that sales after month november increases drastically. This is very clear that in December monthdue to Christmas Eve and New year celebration everone is buying. So sales of Rossmann store is very high in December."""

plt.figure(figsize=(15,6))
sns.pointplot(x= 'DayOfWeek', y= 'Sales', data=final1)
plt.title('Plot between Sales and Day of Week')

"""Plot between Sales and Days of week shows that maximum sales is on Monday and sales gradually decreasing to 6th day of week i.e. on Saturday. It also shows that sales on Sunday is almost near to zero as on sunday maximum stores are closed.

## BoxPlot of sales between Assortment and store type
"""


plt.figure(figsize=(12, 8))
plot_storetype_sales = sns.boxplot(x="StoreType", y="Sales", data=final1)
plt.title('Boxplot For Sales Values')

plt.figure(figsize=(12, 8))
plot_storetype_sales = sns.boxplot(x="Assortment", y="Sales", data=final1)
plt.title('Boxplot For Sales Values on the basis of Assortment Level')

"""### Plot between **Dayof Week** and **Open & promo**."""

sns.countplot(x= 'DayOfWeek', hue='Open', data= final1, palette='cool')
plt.title('Store Daily Open Countplot')

sns.countplot(x= 'DayOfWeek', hue='Promo', data= final1, palette='cool')
plt.title('Store Daily Promo Countplot')

"""## Promo"""

promo_sales = sns.barplot(x="Promo", y="Sales", data=final1, palette='RdPu')

"""Barplot between promo and Sales shows the effect of promotion on Sales. Here 0 represents the store which didnt opt for promotion and 1 represents for stores who opt for promotion. Those store who took promotions their sales are high as compared to stores who didnt took promotion.

## StateHoliday and SchoolHoliday

Sales during State Holiday

0 = public holiday, 1 = Easter holiday, 2 = Christmas, 3 = None
"""

stateholiday_sales = sns.barplot(x="StateHoliday", y="Sales", data=final1)

"""Sales during school holiday"""

schoolholiday_sales = sns.barplot(x="SchoolHoliday", y="Sales", data=final1)

"""We can observe that most of the stores remain closed during State and Holidays. But it is interesting to note that the number of stores opened during School Holidays were more than that were opened during State Holidays.
Another important thing to note is that the stores which were opened during School holidays had more sales than normal.

## Store Type
"""

merged_df = pd.merge(rossmann_df, store_new, on='Store', how='left')

import itertools
fig, axes = plt.subplots(2, 2,figsize=(17,10) )
palette = itertools.cycle(sns.color_palette(n_colors=4))
plt.subplots_adjust(hspace = 0.28)
axes[0,0].bar(merged_df.groupby(by="StoreType").count().Store.index ,merged_df.groupby(by="StoreType").count().Store,color=[next(palette),next(palette),next(palette),next(palette)])
axes[0,0].set_title("Number of Stores per Store Type \n Fig 1.1")
axes[0,1].bar(merged_df.groupby(by="StoreType").sum().Store.index,merged_df.groupby(by="StoreType").sum().Sales/1e9,color=[next(palette),next(palette),next(palette),next(palette)])
axes[0,1].set_title("Total Sales per Store Type \n Fig 1.2")
axes[1,0].bar(merged_df.groupby(by="StoreType").sum().Customers.index,merged_df.groupby(by="StoreType").sum().Customers/1e6,color=[next(palette),next(palette),next(palette),next(palette)])
axes[1,0].set_title("Total Number of Customers per Store Type (in Millions) \n Fig 1.3")
axes[1,1].bar(merged_df.groupby(by="StoreType").sum().Customers.index,merged_df.groupby(by="StoreType").Sales.mean(),color=[next(palette),next(palette),next(palette),next(palette)])
axes[1,1].set_title("Average Sales per Store Type \n Fig 1.4")
plt.show()

"""From this training set we can see that Storetype A has the highest number of branches,sales and customers from the 4 different storetypes. But this doesn't mean it's the best performing Storetype.

When looking at the average sales and number of customers, we see that actually it is Storetype B who was the highest average Sales and highest average Number of Customers.

### Assortments

As we cited in the description, assortments have three types and each store has a defined type and assortment type:

1. a means basic things
2. b means extra things
3. c means extended things so the highest variety of products.
"""

Storetype_Assortment = sns.countplot(x="StoreType",hue="Assortment",order=["a","b","c","d"], data=merged_df,palette=sns.color_palette(n_colors=3)).set_title("Number of Different Assortments per Store Type")
merged_df.groupby(by=["StoreType","Assortment"]).Assortment.count()


## Correlation
"""

numeric_features = ['DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Promo2SinceWeek',
                    'CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear',
                    'Promo2','Promo2SinceWeek','Promo2SinceYear']

for col in numeric_features[0:-1]:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = final1[col]
    label = final1['Sales']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel(col)
    plt.ylabel('Sales')
    ax.set_title('Sales vs ' + col + '- correlation: ' + str(correlation))
    z = np.polyfit(final1[col], final1['Sales'], 1)
    y_hat = np.poly1d(z)(final1[col])

    plt.plot(final1[col], y_hat, "r--", lw=1)

plt.show()

## Correlation
plt.figure(figsize=(18,8))
correlation = final1.corr()
sns.heatmap(abs(correlation), annot=True, cmap='coolwarm')

"""## Multicollinearity"""

from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

calc_vif(final1[[i for i in final1.describe().columns if i not in ['Sales']]])

"""Multicolinearity of columns like 'Promo2SinceYear' is pretty high so we decided to drop it"""

calc_vif(final1[[i for i in final1.describe().columns if i not in ['Sales','Promo2SinceYear']]])

"""Now for each feature VIF values below 10. That's look pretty fine.

## Analysis on Sales - Dependent variable
"""

pd.Series(final1['Sales']).hist()
plt.show()

"""Now checking for number of sales =0."""

final1[(final1.Open == 0) & (final1.Sales == 0)].count()[0]

""" We see that **172817** times store is were temporarily closed for refurbishment. The best solution here is to get rid of closed stores and prevent the models to train on them and get false guidance"""

new_df = final1.drop(final1[(final1.Open == 0) & (final1.Sales == 0)].index)

new_df.head()

new_df.shape

"""PromoInterval to be changed into dummies as it is categorical feature."""

new_df = pd.get_dummies(new_df, columns=['PromoInterval'])

new_df.head()

"""# MODEL TRAINING"""

from scipy.stats import zscore
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score as r2, mean_squared_error as mse
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix,classification_report

"""## MODEL 1 (excluding rows which has sales =0)

We were confused about whether to include rows where sales value is 0.So first we built a model excluding sales value and then including those values
"""

# defining dependent variable
dependent_variables = 'Sales'

# defining independent variable
independent_variables = list(new_df.columns.drop(['Promo2SinceYear','Date','Sales']))

independent_variables

# Create the data of independent variables
X = new_df[independent_variables].values

# Create the data of dependent variable
y = new_df[dependent_variables].values

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)

"""### Linear Regression"""

reg = LinearRegression().fit(X_train, y_train)

reg.score(X_train, y_train)

reg.coef_

reg.intercept_

y_pred = reg.predict(X_test)
y_pred

y_pred_train = reg.predict(X_train)
y_pred_train

y_test

y_train

from sklearn.metrics import mean_squared_error

MSE  = mean_squared_error(y_test, y_pred)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 :" ,r2)

"""### LASSO"""

L1 = Lasso(alpha = 0.2, max_iter=10000)

L1.fit(X_train, y_train)

y_pred_lasso = L1.predict(X_test)

L1.score(X_test, y_test)

pd.DataFrame(zip(y_test, y_pred_lasso), columns = ['actual', 'pred'])

"""### RIDGE"""

L2 = Ridge(alpha = 0.5)

L2.fit(X_train, y_train)

L2.predict(X_test)

L2.score(X_test, y_test)

"""### DECISION TREE"""

decision_tree=DecisionTreeRegressor(max_depth=5)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
y_train_dt = decision_tree.predict(X_train)
#print('dt_regressor R^2: ', r2(v_test,v_pred))
MSE  = mean_squared_error(y_test, y_pred_dt)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)

#RMPSE=RMSE/sales_mean
#print("RMPSE :",RMPSE)

r2 = r2_score(y_test, y_pred_dt)
print("R2 :" ,r2)

"""***

## MODEL 2 (By taking whole Dataset)

We use dummy variables for the column 'PromoInterval'
"""

final1 = pd.get_dummies(final1, columns=['PromoInterval'])

final1.head()

final1.shape

"""We define dependent and independent variables and convert them into arrays"""

# defining dependent variable
dep_var = 'Sales'

# defining independent variable
indep_var = final1.columns.drop(['Store', 'Promo2SinceYear','Date','Sales'])

# Create the data of independent variables
U = final1[indep_var].values

# Create the dependent variable data
v = final1[dep_var].values

final1[indep_var]

"""**We do a train test split keeping the test size as 0.25**"""

# splitting the dataset
U_train, U_test, v_train, v_test = train_test_split(U, v, test_size=0.15, random_state = 0)
print(U_train.shape)
print(U_test.shape)

"""### LINEAR REGRESSION"""

# scling the x values
scaler=StandardScaler()

U_train = scaler.fit_transform(U_train)
U_test = scaler.transform(U_test)

# fitting the data into Lineat Regression Model
linear_regression = LinearRegression()
linear_regression.fit(U_train, v_train)

v_pred=linear_regression.predict(U_test)
v_pred

linear_regression.score(U_train, v_train)

regression_Dataframe = pd.DataFrame(zip(v_test, v_pred), columns = ['actual', 'pred'])
regression_Dataframe

sales_mean=final1[dep_var].mean()

from sklearn.metrics import mean_squared_error

MSE  = mean_squared_error(v_test, v_pred)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)

RMPSE=RMSE/sales_mean
print("RMPSE :",RMPSE)

r2 = r2_score(v_test, v_pred)
print("R2 :" ,r2)

"""### DECISION TREE"""

decision_tree=DecisionTreeRegressor(max_depth=5)
decision_tree.fit(U_train, v_train)
v_pred_dt = decision_tree.predict(U_test)
v_train_dt = decision_tree.predict(U_train)

#print('dt_regressor R^2: ', r2(v_test,v_pred))
MSE_dt  = mean_squared_error(v_test, v_pred_dt)
print("MSE :" , MSE_dt)

RMSE_dt = np.sqrt(MSE_dt)
print("RMSE :" ,RMSE_dt)

RMPSE_dt = RMSE_dt/sales_mean
print("RMPSE :",RMPSE_dt)

R2_dt = r2_score(v_test, v_pred_dt)
print("R2 :" ,R2_dt)

decisiontree_Dataframe = pd.DataFrame(zip(v_test, v_pred_dt), columns = ['actual', 'pred'])
decisiontree_Dataframe

"""### RANDOM FOREST"""

random_forest=RandomForestRegressor(n_estimators =500,max_depth=4)
random_forest.fit(U_train, v_train)
v_pred_rf=random_forest.predict(U_test)
MSE  = mean_squared_error(v_test, v_pred_rf)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)

RMPSE=RMSE/sales_mean
print("RMPSE :",RMPSE)

r2 = r2_score(v_test, v_pred_rf)
print("R2 :" ,r2)

rf_Dataframe = pd.DataFrame(zip(v_test, v_pred_rf), columns = ['actual', 'pred'])
rf_Dataframe

"""### Lasso"""

lasso = Lasso(alpha = 2.0)

lasso.fit(U_train, v_train)

v_pred_lasso = lasso.predict(U_test)

lasso.score(U_train, v_train)

pd.DataFrame(zip(v_test, v_pred_lasso), columns = ['actual', 'pred'])

"""### Ridge"""

ridge = Ridge(alpha = 0.5)

ridge.fit(U_train, v_train)

v_pred_rid=ridge.predict(U_test)

ridge.score(U_test, v_test)

MSE  = mean_squared_error(v_test, v_pred_rid)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)

RMPSE=RMSE/sales_mean
print("RMPSE :",RMPSE)

r2 = r2_score(v_test, v_pred_rid)
print("R2 :" ,r2)




