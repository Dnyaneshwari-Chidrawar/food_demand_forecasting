#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[633]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, r2_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


from xgboost import XGBRegressor


# ## Understanding the data

# In[634]:


directory_path_1 = '/home/pc/Desktop_linux/chinu/food_demand_forecasting/train_GzS76OK'
directory_path_2 = '/home/pc/Desktop_linux/chinu/food_demand_forecasting'

file_name_1 = 'meal_info.csv'
file_name_2 = 'fulfilment_center_info.csv'
file_name_3 = 'train.csv'
test_file_name = 'test_QoiMO9B.csv'

file_1 = os.path.join(directory_path_1, file_name_1)
file_2 = os.path.join(directory_path_1, file_name_2)
file_3 = os.path.join(directory_path_1, file_name_3)
test_file = os.path.join(directory_path_2, test_file_name)

meal = pd.read_csv(file_1)
center = pd.read_csv(file_2)
train_df = pd.read_csv(file_3)
unseen_df = pd.read_csv(test_file)

print(meal.head())
print(center.head())
print(train_df)
print(unseen_df)


# In[635]:


center['region_code'] = center['region_code'].astype('category')


# In[636]:


train_df.shape, center.shape, meal.shape, unseen_df.shape


# In[637]:


train_df.center_id.nunique(), train_df.meal_id.nunique()


# In[638]:


plt.figure(figsize=(3,3))
plt.scatter(x=train_df.index, y=train_df['num_orders'])
plt.show()

plt.figure(figsize=(3,3))
plt.hist(train_df['num_orders'], bins=40)
plt.show()


# In[639]:


# There is no missing values

print(train_df.isna().sum())
print(meal.isna().sum())
print(center.isna().sum())
print(unseen_df.isna().sum())


# ## Merging the tables

# In[640]:


train_df = pd.merge(train_df, center, how='left', on='center_id')
train_df = pd.merge(train_df, meal, how='left', on='meal_id')

unseen_df = pd.merge(unseen_df, center, how='left', on='center_id')
unseen_df = pd.merge(unseen_df, meal, how='left', on='meal_id')

train_df.head(), unseen_df.head()


# In[641]:


train_df.info()


# In[642]:


unseen_df.info()


# ## Dividing the feature into categorical or numerical

# In[643]:


# for feature in train_df.columns:
#     print(feature)
#     plt.figure(figsize=(4,4))
#     plt.scatter(train_df['id'], y=train_df[feature])
#     plt.show()


# ## Visualising the data

# In[644]:


plt.figure(figsize=(20,20))
plt.subplot(3,3,1)
sns.boxplot(data=train_df, x='region_code', y='num_orders')

plt.subplot(3,3,2)
sns.boxplot(data=train_df, x='emailer_for_promotion', y='num_orders')

plt.subplot(3,3,3)
sns.boxplot(data=train_df, x='homepage_featured', y='num_orders')

plt.subplot(3,3,4)
sns.boxplot(data=train_df, x='city_code', y='num_orders')

plt.subplot(3,3,5)
sns.boxplot(data=train_df, x='center_type', y='num_orders')

plt.subplot(3,3,6)
sns.boxplot(data=train_df, x='category', y='num_orders')

plt.subplot(3,3,7)
sns.boxplot(data=train_df, x='cuisine', y='num_orders')


# ### Insight:  
# - there is no missing values.  

# In[645]:


numeric = [ 'checkout_price', 'base_price',  'op_area','num_orders']
categorical = ['center_id', 'meal_id', 'emailer_for_promotion', 'homepage_featured', 'city_code', 'region_code', 'center_type',  'category', 'cuisine']


# ## Correlation

# In[646]:


sns.heatmap(train_df[numeric].corr(), annot=True)
plt.show()


# ## Insight:  
# - There is a high correlation between 'checkout_price' and 'base_price'. 
# - I should take only one variable in regression.

# ## Plots of numeric variable

# In[647]:


for feature in numeric:
    plt.figure(figsize=(2,2))
    plt.hist(train_df[feature], bins=20, density=True)
    # sns.histplot(train_df_2[feature])
    plt.title(feature)
    plt.show()


# ## Insight:
# - choosing 'checkout_price' feature over 'base_price' as it is comparitivly more normally distributed.  
# 

# ## One hot encoding for categorical

# In[648]:


one_hot_columns = ['emailer_for_promotion', 'homepage_featured', 'region_code', 'center_type', 'category', 'cuisine']
dummy_df = pd.get_dummies(train_df[one_hot_columns])
train_df = pd.concat([train_df, dummy_df], axis=1)

dummy_df = pd.get_dummies(unseen_df[one_hot_columns])
unseen_df = pd.concat([unseen_df, dummy_df], axis=1)


# ## Checking unseen_df and train_df have same columns

# In[649]:


for feature_train in train_df.columns:
    if feature_train not in unseen_df.columns:
        print(feature_train)


for feature in unseen_df.columns:
    if feature not in train_df.columns:
        print(feature)


# 
# ## Assign scores to center_id

# In[650]:


center_df = train_df.groupby(by='center_id')['num_orders'].sum().reset_index()
scaler = MinMaxScaler()
center_df['center_score'] = scaler.fit_transform(center_df[['num_orders']])
center_df['center_score'] = center_df['center_score'].apply(lambda x: round(x, 3))

train_df = train_df.merge(center_df[['center_id', 'center_score']], how='left', on='center_id')
train_df.head()


# In[651]:


unseen_df = unseen_df.merge(center_df[['center_id', 'center_score']], how='left', on='center_id')
unseen_df.head()


# ## Assign scores to meal_id

# In[652]:


meal_df = train_df.groupby(by='meal_id')[['num_orders']].sum().reset_index()
scaler = MinMaxScaler()
meal_df['meal_score'] = scaler.fit_transform(meal_df[['num_orders']])
meal_df['meal_score'] = meal_df['meal_score'].apply(lambda x: round(x, 3))
train_df = train_df.merge(meal_df[['meal_id', 'meal_score']], how='left', on='meal_id')
train_df.head()


# In[653]:


unseen_df = unseen_df.merge(meal_df[['meal_id', 'meal_score']], how='left', on='meal_id')
unseen_df.head()


# ## Assign scores to city code feature

# In[656]:


city_df = train_df.groupby(by='city_code')[['num_orders']].sum().reset_index()
scaler = MinMaxScaler()
city_df['city_score'] = scaler.fit_transform(city_df[['num_orders']])
city_df['city_score'] = city_df['city_score'].apply(lambda x: round(x, 3))
train_df = train_df.merge(city_df[['city_code', 'city_score']], how='left', on='city_code')
train_df.head()


# In[657]:


unseen_df = unseen_df.merge(city_df[['city_code', 'city_score']], how='left', on='city_code')
unseen_df.head()


# ## Drop columns

# In[658]:


drop_columns = ['id', 'week', 'center_id', 'meal_id', 'city_code',
                'base_price', 'emailer_for_promotion', 'homepage_featured',
                'region_code', 'center_type', 'category', 'cuisine']
train_df = train_df.drop(columns=drop_columns)
unseen_df = unseen_df.drop(columns=drop_columns)


# ## Train - Test split

# In[659]:


y = train_df[['num_orders']]
X = train_df.drop(columns='num_orders')
y.shape, X.shape


# In[660]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[661]:


unseen_df.shape


# ## Model Evaluation

# In[662]:


def model_evaluation(y_train, y_test, y_train_pred, y_test_pred, y_train_pred_clip, y_test_pred_clip, model='model'):
    print('* ' * 15, f'{model}', ' *' * 15)
    print(" For Train Data : ")
    msle_train = mean_squared_log_error(y_train, y_train_pred_clip)
    print('RMSLE = ', msle_train)

    r2_score_train = r2_score(y_train, y_train_pred)
    print("R2 Score  = ", r2_score_train)

    print(" ")
    print("\n For Test Data : ")
    msle_test = mean_squared_log_error(y_test, y_test_pred_clip)
    print('RMSLE = ', msle_test)

    r2_score_test = r2_score(y_test, y_test_pred)
    print("R2 Score  = ", r2_score_test)


# ## Linear Regression

# In[663]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[664]:


y_train_pred = reg.predict(X_train).astype(int)
y_test_pred = reg.predict(X_test).astype(int)

y_train_pred_clip = np.clip(y_train_pred, 0, None)
y_test_pred_clip = np.clip(y_test_pred, 0, None)
model_evaluation(y_train, y_test, y_train_pred, y_test_pred, y_train_pred_clip, y_test_pred_clip, reg)


# ## Random Forest

# In[665]:


param = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, 20, 25],
    # 'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6]
}

msle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)

model = RandomForestRegressor()
grid = GridSearchCV(estimator=model,
                    param_grid=param,
                    cv=4,
                    scoring=msle_scorer,
                    verbose=2,
                    n_jobs=-1)

grid.fit(X_train, y_train.squeeze())
print(grid.best_estimator_)


# In[667]:


print(grid.best_estimator_, grid.best_score_)


# In[668]:


random1 = RandomForestRegressor(max_depth=20, min_samples_leaf=4, n_estimators=150, random_state=42)
random1.fit(X_train, y_train.squeeze())


# In[669]:


y_train_pred = random1.predict(X_train).astype(int)
y_test_pred = random1.predict(X_test).astype(int)
unseen_pred = random1.predict(unseen_df).astype(int)

y_train_pred_clip = np.clip(y_train_pred, 0, None)
y_test_pred_clip = np.clip(y_test_pred, 0, None)
unseen_pred_clip = np.clip(unseen_pred, 0, None)
model_evaluation(y_train, y_test, y_train_pred, y_test_pred, y_train_pred_clip, y_test_pred_clip, random1)


# ## XG Boosting

# In[ ]:


xgb = XGBRegressor()
param ={
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    # 'gamma': [0, 1, 5],
    # 'reg_alpha': [0, 0.1, 1],
    # 'reg_lambda': [1, 2, 5]
}

scoring = make_scorer(mean_squared_log_error)
# scoring = make_scorer(mean_squared_log_error, greater_is_better=False)

rand_search = RandomizedSearchCV(
                                    estimator=xgb,
                                    param_distributions=param,
                                    scoring=scoring,
                                    cv=3,          # 3-fold cross-validation
                                    verbose=2,     # Show progress
                                    n_jobs=-1      # Use all available CPU cores
                                )

rand_search.fit(X_train, y_train)


# ## Submission file

# In[670]:


unseen_df = pd.read_csv(test_file)
submission_df = unseen_df[['id']]
submission_df['num_orders'] = unseen_pred_clip
submission_df.head(), submission_df.shape, submission_df.columns


# In[671]:


file_name = 'Output_random_forest_3.csv'
submission_file = os.path.join(directory_path_2, file_name)
submission_df.to_csv(submission_file, index=False)

# Model Name
# random1 = RandomForestRegressor(max_depth=20, min_samples_leaf=4, n_estimators=150, random_state=42)


# In[ ]:





# In[631]:


import nbconvert

# Specify the notebook file
file_1 = "forecasting.ipynb"
notebook_filename = os.path.join(directory_path_2, file_1)

# Convert the notebook to a Python script
exporter = nbconvert.PythonExporter()
script, _ = exporter.from_filename(notebook_filename)

# Save the script to a file
file_1 = "forecasting_submission_2.py"
notebook_filename = os.path.join(directory_path_2, file_1)
with open(notebook_filename, "w") as f:
    f.write(script)


# In[ ]:




