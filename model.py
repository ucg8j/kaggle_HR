import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from random import sample

#### set working dir and read data
path = os.path.expanduser('~/Projects/kaggle_HR/')
os.chdir(path)
df = pd.read_csv('HR_comma_sep.csv')


#### explore the data set
col_names = df.columns.tolist()
df.shape # (14999, 10)
df.describe

# expand possible variables
df['productivity'] = df.average_montly_hours / df.number_projectdf

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

#### TODO alternative split, Cross Validate


#### model 
# two class forest

# two class jungle
# two class logistic regression
# two class neural net
# two class decision tree
# two class bayes

#### evaluate


