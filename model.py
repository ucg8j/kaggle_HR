import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import tree
from IPython.display import Image  
from pydotplus import graph_from_dot_data

#### set working dir and read data
path = os.path.expanduser('~/Projects/kaggle_HR/')
os.chdir(path)
df = pd.read_csv('HR_comma_sep.csv')

np.random.seed(42)

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

# TODO interaction terms

# some basic qtys
print '# people left = {}'.format(len(df[df['left'] == 1]))
# people left = 3571
print '# people stayed = {}'.format(len(df[df['left'] == 0]))
# people stayed = 11428
print '% people left = {}%'.format(round(float(len(df[df['left'] == 1])) / len(df) * 100), 3)
# % people left = 24.0%

# check missing for values (none)
df.apply(lambda x: sum(x.isnull()), axis=0)

# # correlation heatmap
# correlation = df.corr()
# plt.figure(figsize=(10, 10))
# sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

# # pairwise plots
# df_sample = df.sample(frac=0.05)
# pplot = sns.pairplot(df_sample, hue="left")
# pplot_v2 = sns.pairplot(df_sample, diag_kind="kde")

# TODO check for outliers

#### split data into train, test and validate
# 60% - train set, 20% - validation set, 20% - test set
train, test, validate = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
print train.shape, test.shape, validate.shape

# Separate target and predictors
y_train = train['left']
x_train = train.drop(['left'], axis=1)
y_test = test['left']
x_test = test.drop(['left'], axis=1)
y_validate = validate['left']
x_validate = validate.drop(['left'], axis=1)

#### variable importance / selection
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True)
#[(0.3329, 'satisfaction_level'), (0.1751, 'time_spend_company'), (0.1389, 'average_montly_hours'), (0.1285, 'number_project'), (0.1052, 'last_evaluation'), (0.0836, 'productivity'), (0.0066, 'Work_accident'), (0.0045, '$_medium'), (0.004, '$_high'), (0.0034, 'sales_technical'), (0.0031, 'sales_sales'), (0.0025, 'sales_support'), (0.0024, 'sales_RandD'), (0.0023, 'sales_hr'), (0.002, 'promotion_last_5years'), (0.0015, 'sales_accounting'), (0.0014, 'sales_marketing'), (0.0013, 'sales_management'), (0.0009, 'sales_product_mng')]

# Assign feature importance and sort
importances = rf.feature_importances_
std = np.std([rf.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot variable importance
plt.figure()
plt.title("Feature importance")
plt.bar(range(x_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")

# Create variable lists and drop
all_vars = x_train.columns.tolist()
top_5_vars = ['satisfaction_level', 'number_project', 'time_spend_company', 
              'average_montly_hours', 'last_evaluation']
bottom_vars = [cols for cols in all_vars if cols not in top_5_vars]

# Drop less important variables leaving the top_5
x_train    = x_train.drop(bottom_vars, axis=1)
x_test     = x_test.drop(bottom_vars, axis=1)
x_validate = x_validate.drop(bottom_vars, axis=1)


#### model
# LOGISTIC REGRESSION
logit_model = LogisticRegression()                # instantiate
logit_model = logit_model.fit(x_train, y_train)   # fit
logit_model.score(x_train, y_train)               # Accuracy 0.768

# examine the coefficients
pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

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

## RANDOM FORREST (ALL Variables)

# Instantiate
rf = RandomForestClassifier()	   
# Fit
rf_model = rf.fit(x_train, y_train)
# Accuracy 99.84%
rf_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(rf_model.predict(x_test))
probs = pd.DataFrame(rf_model.predict_proba(x_test))

# metrics
print metrics.accuracy_score(y_test, predicted)
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
####################################################################################################################################################################################################
## RANDOM FORREST (top_2_vars )

# Instantiate
rf = RandomForestClassifier()
# Fit
rf_model_6vars = rf.fit(x_train[top_2_vars], y_train)
# Accuracy 99.8%
rf_model_6vars.score(x_train[top_2_vars], y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(rf_model_6vars.predict(x_test[top_2_vars]))
probs = pd.DataFrame(rf_model_6vars.predict_proba(x_test[top_2_vars]))

# metrics
print metrics.accuracy_score(y_test, predicted)
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
####################################################################################################################################################################################################

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
scores = cross_val_score(SVC(probability=True), x_test, y_test, scoring='accuracy', cv=10)
print scores.mean() # 0.924998514798

# DECISION TREE (pruned to depth of 5)
# TODO optimise depth
tree_model = tree.DecisionTreeClassifier(max_depth=2) # instantiate
tree_model = tree_model.fit(x_train, y_train)    # fit
tree_model.score(x_train, y_train)               # Accuracy 0.97577508612068009

# predictions/probs on the test dataset
predicted = pd.DataFrame(tree_model.predict(x_test))
probs = pd.DataFrame(tree_model.predict_proba(x_test))

# metrics
print metrics.accuracy_score(y_test, predicted)     # 0.978333333333
print metrics.roc_auc_score(y_test, probs[1])       # 0.976297791362
print metrics.confusion_matrix(y_test, predicted) 
# [[2269   13]
#  [  52  666]]
print metrics.classification_report(y_test, predicted)
#              precision    recall  f1-score   support
#           0       0.98      0.99      0.99      2282
#           1       0.98      0.93      0.95       718
# avg / total       0.98      0.98      0.98      3000

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(DecisionTreeClassifier(), x_test, y_test, scoring='accuracy', cv=10)
print scores.mean() # 0.960658722134

# output decision plot
dot_data = tree.export_graphviz(tree_model, out_file=None, 
                     feature_names=x_test.columns.tolist(),
                     class_names=['remain', 'left'],
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graph_from_dot_data(dot_data)
graph.write_png("decision_tree.png")
# Image(grsacaph.create_png()) # TODO uncomment for notebook.py

# KNN ?
tree_model = tree.DecisionTreeClassifier(max_depth=2) # instantiate
tree_model = tree_model.fit(x_train, y_train)    # fit
# gradient boost
# two class bayes
# two class neural net

#### evaluate
#Model comparison
models = pd.DataFrame({
    'Model'          : ['Logistic Regression', 'SVM', 'kNN', 'Decision Tree', 'Random Forest'],
    'Training_Score' : [logis_score_train, svm_score_train, knn_score_train, dt_score_train, rfc_score_train],
    'Testing_Score'  : [logis_score_test, svm_score_test, knn_score_test, dt_score_test, rfc_score_test]
    })
models.sort_values(by='Testing_Score', ascending=False)

# overfit to particular period?
# tree better than SVM considering interpretability in bus context and training time