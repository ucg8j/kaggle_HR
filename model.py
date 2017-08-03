import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
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
pd.DataFrame(zip(x_train.columns, np.transpose(logit_model.coef_)))

# predictions on the test dataset
predicted = pd.DataFrame(logit_model.predict(x_test))
print predicted.head(n=15)

# probabilities on the test dataset
probs = pd.DataFrame(logit_model.predict_proba(x_test))
print probs.head(n=15)
# if the probability of an employee is >50% then prediction will be =1

# Store metrics
logit_accuracy = metrics.accuracy_score(y_test, predicted)     
logit_roc_auc = metrics.roc_auc_score(y_test, probs[1])       
logit_confus_matrix = metrics.confusion_matrix(y_test, predicted) 
logit_classification_report = metrics.classification_report(y_test, predicted)
logit_precision = metrics.precision_score(y_test, predicted, pos_label=1)
logit_recall = metrics.recall_score(y_test, predicted, pos_label=1)
logit_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# evaluate the model using 10-fold cross-validation
logit_cv_scores = cross_val_score(LogisticRegression(), x_test, y_test, scoring='precision', cv=10)
logit_cv_mean = np.mean(logit_cv_scores)

# DECISION TREE (pruned to depth of 3)
# TODO optimise depth
# Instantiate with a max depth of 3
tree_model = tree.DecisionTreeClassifier(max_depth=3) 
# Fit a decision tree
tree_model = tree_model.fit(x_train, y_train)
# Training accuracy
tree_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(tree_model.predict(x_test))
probs = pd.DataFrame(tree_model.predict_proba(x_test))

# Store metrics
tree_accuracy = metrics.accuracy_score(y_test, predicted)     
tree_roc_auc = metrics.roc_auc_score(y_test, probs[1])       
tree_confus_matrix = metrics.confusion_matrix(y_test, predicted) 
tree_classification_report = metrics.classification_report(y_test, predicted)
tree_precision = metrics.precision_score(y_test, predicted, pos_label=1)
tree_recall = metrics.recall_score(y_test, predicted, pos_label=1)
tree_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# evaluate the model using 10-fold cross-validation
tree_cv_scores = cross_val_score(tree.DecisionTreeClassifier(max_depth=3), 
                                x_test, y_test, scoring='precision', cv=10)
tree_cv_mean = np.mean(tree_cv_scores)

# output decision plot
dot_data = tree.export_graphviz(tree_model, out_file=None, 
                     feature_names=x_test.columns.tolist(),
                     class_names=['remain', 'left'],
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graph_from_dot_data(dot_data)
graph.write_png("images/decision_tree.png")


## RANDOM FOREST
# Instantiate
rf = RandomForestClassifier()	   
# Fit
rf_model = rf.fit(x_train, y_train)
# training accuracy 99.74%
rf_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(rf_model.predict(x_test))
probs = pd.DataFrame(rf_model.predict_proba(x_test))

# Store metrics
rf_accuracy = metrics.accuracy_score(y_test, predicted)     
rf_roc_auc = metrics.roc_auc_score(y_test, probs[1])       
rf_confus_matrix = metrics.confusion_matrix(y_test, predicted) 
rf_classification_report = metrics.classification_report(y_test, predicted)
rf_precision = metrics.precision_score(y_test, predicted, pos_label=1)
rf_recall = metrics.recall_score(y_test, predicted, pos_label=1)
rf_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# evaluate the model using 10-fold cross-validation
rf_cv_scores = cross_val_score(RandomForestClassifier(), x_test, y_test, scoring='precision', cv=10)
rf_cv_mean = np.mean(rf_cv_scores)


# SUPPORT VECTOR MACHINE
# Instantiate
svm_model = SVC(probability=True)
# Fit
svm_model = svm_model.fit(x_train, y_train)
# Accuracy
svm_model.score(x_train, y_train)

# predictions/probs on the test dataset
predicted = pd.DataFrame(svm_model.predict(x_test))
probs = pd.DataFrame(svm_model.predict_proba(x_test))

# Store metrics
svm_accuracy = metrics.accuracy_score(y_test, predicted)     
svm_roc_auc = metrics.roc_auc_score(y_test, probs[1])       
svm_confus_matrix = metrics.confusion_matrix(y_test, predicted) 
svm_classification_report = metrics.classification_report(y_test, predicted)
svm_precision = metrics.precision_score(y_test, predicted, pos_label=1)
svm_recall = metrics.recall_score(y_test, predicted, pos_label=1)
svm_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# evaluate the model using 10-fold cross-validation
svm_cv_scores = cross_val_score(SVC(probability=True), x_test, y_test, scoring='precision', cv=10)
svm_cv_mean = np.mean(svm_cv_scores)


# KNN
# instantiate learning model (k = 3)
knn_model = KNeighborsClassifier(n_neighbors=3)
# fit the model
knn_model.fit(x_train, y_train)
# Accuracy
knn_model.score(x_train, y_train)

# predictions/probs on the test dataset
predicted = pd.DataFrame(knn_model.predict(x_test))
probs = pd.DataFrame(knn_model.predict_proba(x_test))

# Store metrics
knn_accuracy = metrics.accuracy_score(y_test, predicted)     
knn_roc_auc = metrics.roc_auc_score(y_test, probs[1])       
knn_confus_matrix = metrics.confusion_matrix(y_test, predicted) 
knn_classification_report = metrics.classification_report(y_test, predicted)
knn_precision = metrics.precision_score(y_test, predicted, pos_label=1)
knn_recall = metrics.recall_score(y_test, predicted, pos_label=1)
knn_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# evaluate the model using 10-fold cross-validation
knn_cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), x_test, y_test, scoring='precision', cv=10)
knn_cv_mean = np.mean(knn_cv_scores)


# TWO CLASS BAYES

# instantiate
bayes_model = GaussianNB()
# fit the model
bayes_model.fit(x_train, y_train)
# Accuracy
bayes_model.score(x_train, y_train)

# predictions/probs on the test dataset
predicted = pd.DataFrame(bayes_model.predict(x_test))
probs = pd.DataFrame(bayes_model.predict_proba(x_test))

# Store metrics
bayes_accuracy = metrics.accuracy_score(y_test, predicted)     
bayes_roc_auc = metrics.roc_auc_score(y_test, probs[1])       
bayes_confus_matrix = metrics.confusion_matrix(y_test, predicted) 
bayes_classification_report = metrics.classification_report(y_test, predicted)
bayes_precision = metrics.precision_score(y_test, predicted, pos_label=1)
bayes_recall = metrics.recall_score(y_test, predicted, pos_label=1)
bayes_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

# evaluate the model using 10-fold cross-validation
bayes_cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), x_test, y_test, scoring='precision', cv=10)
bayes_cv_mean = np.mean(bayes_cv_scores)


#### evaluate
# Model comparison
models = pd.DataFrame({
    'Model'    : ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'kNN', 'Bayes'],
    'Accuracy' : [logit_accuracy, tree_accuracy, rf_accuracy, svm_accuracy, knn_accuracy, bayes_accuracy],
    'Precision': [logit_precision, tree_precision, rf_precision, svm_precision, knn_precision, bayes_precision],
    'recall'   : [logit_recall, tree_recall, rf_recall, svm_recall, knn_recall, bayes_recall],
    'F1'       : [logit_f1, tree_f1, rf_f1, svm_f1, knn_f1, bayes_f1],
    'cv_precision' : [logit_cv_mean, tree_cv_mean, rf_cv_mean, svm_cv_mean, knn_cv_mean, bayes_cv_mean]
    })
models.sort_values(by='Precision', ascending=False)

#### Save model
import cPickle
with open('churn_classifier.pkl', 'wb') as fid:
    cPickle.dump(rf_model, fid)
