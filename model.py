import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm # logistic but... use 
from random import sample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.svm import SVC


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

# TODO - dummy ALTERNATIVE
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

# TODO interaction terms
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
plt.figure(figsize=(10, 10))
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


#### model
# LOGISTIC REGRESSION
logit_model = LogisticRegression()                # instantiate
logit_model = logit_model.fit(x_train, y_train)   # fit
logit_model.score(x_train, y_train)               # Accuracy 80%

# what % of the test set leave?
y_test.mean()
# 0.23933333333333334 
# you could obtain 76% accuracy by always predicting 'left = N' (i.e. remain in org). 
# In that sense we are doing that well!

# examine the coefficients
pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
#    satisfaction_level     [-3.93663051447] lower satisfaction associated with leaving
#       last_evaluation     [0.689767457149] higher performance associated with leaving
#        number_project     [0.110208280701] more projects (performance related) associated with leaving
#  average_montly_hours  [-0.00408269509067]
#    time_spend_company     [0.263577652969]
#         Work_accident     [-1.56599235287]
# promotion_last_5years     [-1.24075021228]
#                $_high     [-1.89336653791]
#              $_medium    [-0.523404690533]
#           sales_RandD    [-0.497071456721]
#      sales_accounting    [0.0938805031077]
#              sales_hr     [0.331180487373]
#      sales_management    [-0.510669271677]
#       sales_marketing    [0.0420679846999]
#     sales_product_mng    [0.0389480539297]
#           sales_sales    [0.0978587842515]
#         sales_support      [0.15181336107]
#       sales_technical     [0.187623110255]
#          productivity    [0.0276848033542]

# predictions on the test dataset
predicted = pd.DataFrame(logit_model.predict(x_test))
print predicted.head(n=15)

# probabilities on the test dataset
probs = pd.DataFrame(logit_model.predict_proba(x_test))
print probs.head(n=15)
# if the probability of an employee is >50% then prediction will be =1

print metrics.accuracy_score(y_test, predicted)     # 0.803
print metrics.roc_auc_score(y_test, probs[1])       # 0.822072462459
print metrics.confusion_matrix(y_test, predicted) 
# [[2123  159]
#  [ 432  286]]
print metrics.classification_report(y_test, predicted)
#              precision    recall  f1-score   support
#           0       0.83      0.93      0.88      2282
#           1       0.64      0.40      0.49       718
# avg / total       0.79      0.80      0.79      3000

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), x_test, y_test, scoring='accuracy', cv=10)
print scores.mean()

# RANDOM FORREST
rf_model = rf.fit(x_train, y_train)   # fit
rf_model.score(x_train, y_train)      # Accuracy 99.8%
# much better than logit_model maybe overfit though

# predictions/probs on the test dataset
predicted = pd.DataFrame(rf_model.predict(x_test))
probs = pd.DataFrame(rf_model.predict_proba(x_test))

# metrics
print metrics.accuracy_score(y_test, predicted)     # 0.803
print metrics.roc_auc_score(y_test, probs[1])       # 0.990297386108
print metrics.confusion_matrix(y_test, predicted) 
# [[2280    2]
#  [  35  683]]
print metrics.classification_report(y_test, predicted)
#              precision    recall  f1-score   support
#           0       0.98      1.00      0.99      2282
#           1       1.00      0.95      0.97       718
# avg / total       0.99      0.99      0.99      3000

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(RandomForestClassifier(), x_test, y_test, scoring='accuracy', cv=10)
print scores.mean() # 0.976325455468

# TODO investigate overfitting/model with less vars/improve generalisation

# SUPPORT VECTOR MACHINE
svm_model = SVC(probability=True)			   # instantiate
svm_model = svm_model.fit(x_train, y_train)    # fit
svm_model.score(x_train, y_train)              # Accuracy 95%

# predictions/probs on the test dataset
predicted = pd.DataFrame(svm_model.predict(x_test))
probs = pd.DataFrame(svm_model.predict_proba(x_test))

# metrics
print metrics.accuracy_score(y_test, predicted)     # 0.937666666667
print metrics.roc_auc_score(y_test, probs[1])       # 0.967936667977
print metrics.confusion_matrix(y_test, predicted) 
# [[2181  101]
#  [  86  632]]
print metrics.classification_report(y_test, predicted)
#              precision    recall  f1-score   support
#           0       0.96      0.96      0.96      2282
#           1       0.86      0.88      0.87       718
# avg / total       0.94      0.94      0.94      3000

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(RandomForestClassifier(), x_test, y_test, scoring='accuracy', cv=10)
print scores.mean() # 0.977325455468

# two class decision tree
# two class bayes
# two class neural net
#### evaluate
# overfit to particular period?