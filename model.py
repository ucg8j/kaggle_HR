import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from random import sample
from sklearn.ensemble import RandomForestClassifier

#### set working dir and read data
path = os.path.expanduser('~/Projects/kaggle_HR/')
os.chdir(path)
df = pd.read_csv('HR_comma_sep.csv')


#### explore the data set
col_names = df.columns.tolist()
df.shape # (14999, 10)
df.describe

# one hot encode string/'object' variables 
df.dtypes # 'salary' and 'sales'
df['salary'] = '$_' + df['salary'].astype(str)
df['sales'] = 'sales_' + df['sales'].astype(str)

# create salary dummies and join
one_hot_salary = pd.get_dummies(df['salary'])
df = df.join(one_hot_salary)

# create sales dummies and join
one_hot_sales = pd.get_dummies(df['sales'])
df = df.join(one_hot_sales)

# drop unnecessay columns
df = df.drop(['salary', 'sales', '$_low', 'sales_IT'], axis=1)

# ALTERNATIVE
# n.b. python has no native way of handling factors like R and sklearn's rf classifier doesn't 
# handle categorical vars. Therefore need to one hot encode OR convert to numeric representations.
#Auto encodes any dataframe column of type category or object.
# from sklearn.preprocessing import LabelEncoder
# def dummyEncode(df):
#         columnsToEncode = list(df.select_dtypes(include=['category','object']))
#         le = LabelEncoder()
#         for feature in columnsToEncode:
#             try:
#                 df[feature] = le.fit_transform(df[feature])
#             except:
#                 print('Error encoding '+feature)
#         return df
        
# df_test = dummyEncode(df)

# expand possible variables
df['productivity'] = df.average_montly_hours / df.number_project

# some basic qtys
print '# people left = {}'.format(len(df[df['left'] == 1]))
# people left = 3571
print '# people stayed = {}'.format(len(df[df['left'] == 0]))
# people stayed = 11428
print '% people stayed = {}%'.format(round(float(len(df[df['left'] == 1])) / len(df) * 100), 3)
# % people stayed = 24.0%

# check missing for values
df.apply(lambda x: sum(x.isnull()), axis=0)
# no missing values

# correlation heatmap
correlation = df.corr()
plt.figure(figsize = (10,10))
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

# pairwise plots
df_sampl = df.sample(frac=0.05)
pplot = sns.pairplot(df_sampl, hue="left")
pplot_v2 = sns.pairplot(df_sampl, diag_kind="kde")

# TODO check for outliers

#### split data into train, test and validate
# 60% - train set, 20% - validation set, 20% - test set
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

# separate target and predictors
y_train = train['left']
x_train = train.drop(['left'], axis=1)
y_validate = validate['left']
x_validate = validate.drop(['left'], axis=1)
y_test = test['left']
x_test = test.drop(['left'], axis=1)

#### variable importance / selection
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True)

importances = rf.feature_importances_
std = np.std([rf.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# [(0.245, 'satisfaction_level'), (0.1728, 'number_project'), (0.1655, 'time_spend_company'),
# (0.1369, 'average_montly_hours'), (0.1244, 'productivity'), (0.1139, 'last_evaluation'), 
# (0.0111, 'Work_accident'), (0.0063, '$_high'), (0.0043, '$_medium'), (0.0037, 'sales_sales'), 
# (0.0035, 'sales_technical'), (0.0031, 'sales_support'), (0.0019, 'sales_management'), 
# (0.0018, 'sales_hr'), (0.0016, 'sales_RandD'), (0.0015, 'sales_marketing'), 
# (0.0012, 'sales_accounting'), (0.0008, 'promotion_last_5years'), (0.0007, 'sales_product_mng')]

# plot variable importance
plt.figure()
plt.title("Feature importance")
plt.bar(range(x_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")

# create variable lists
all_vars = df.columns.tolist()
top_6_vars = ['satisfaction_level', 'number_project', 'time_spend_company', 
             'average_montly_hours', 'productivity', 'last_evaluation']
top_5_vars = ['satisfaction_level', 'number_project', 'time_spend_company', 
             'average_montly_hours', 'last_evaluation']

#### TODO alternative split, Cross Validate


#### model 
# logistic regression
logit_all_vars = sm.Logit(y_train, x_train)
result = logit_all_vars.fit()
print result.summary()
print np.exp(result.params)

# make predictions on the enumerated dataset
y_test_l = result.predict(x_test)
print("Test accuracy: ", sum(pd.DataFrame(y_test_l).idxmax(axis=1).values == y_test) / float(len(y_test)))
# ('Test accuracy: ', 0.76066666666666671)

y_val_l = result.predict(x_validate)
print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values == y_validate) / float(len(y_validate)))
# ('Validation accuracy: ', 0.75466666666666671)

# two class forest
rf.fit(x_train, y_train)
col_names = x_test.columns.tolist()
test['left_predict_rf'] = rf.predict_proba(test[col_names])


# two class jungle

# two class neural net
# two class decision tree
# two class bayes

#### evaluate


