import pandas as pd
import os

#### set working dir and read data
path = os.path.expanduse('~/Projects/kaggle_HR/')
os.chdir(path)
df = pd.read_csv('HR_comma_sep.csv')


#### explore the data set
columns_names = df.columns.tolist()
df.shape # (14999, 11)


print('# people left = {}'.format(df[df['left']==1].size))
# people left = 39281
print('# people stayed = {}'.format(df[df['left']==0].size))
# of people stayed = 125708
print('% people stayed = {}%'.format(round(float(len(df[df['left']==1])) / len(df) * 100),3))
# % people stayed = 24.0%



#### feature eng / expand possible variables
df['productivity'] = df.average_montly_hours / df.number_project


#### predict

