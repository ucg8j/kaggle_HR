import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#### set working dir and read data
path = os.path.expanduse('~/Projects/kaggle_HR/')
os.chdir(path)
df = pd.read_csv('HR_comma_sep.csv')


#### explore the data set
col_names = df.columns.tolist()
df.shape # (14999, 11)
df.describe

# some basic qtys
print '# people left = {}'.format(len(df[df['left'] == 1]))
# people left = 3571
print '# people stayed = {}'.format(len(df[df['left'] == 0]))
# people stayed = 11428
print '% people stayed = {}%'.format(round(float(len(df[df['left'] == 1])) / len(df) * 100), 3)
# % people stayed = 24.0%

# correlation heatmap
correlation = df.corr()
plt.figure(figsize = (10,10))
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Correlation between numeric features')

# TODO fix this plot
#pplot = sns.pairplot(df)

#### feature eng / expand possible variables
df['productivity'] = df.average_montly_hours / df.number_projectdf


#### predict


