import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('input.csv')
data = data.dropna()
num_cols = data._get_numeric_data().columns

#Numeric columns
['Dexa_Freq_During_Rx', 'Count_Of_Risks']

data.columns = [ x.lower().strip() for x in data.columns]

#for col in data.columns:
#    if col not in num_cols:
#        data[col] = data[col].strip()

unknown_dict = {}

#for col in data.columns:

#    unknown_dict[col] = data[data[col]=='Unknown'].shape[0]

grouping = ['concom','comorb','risk']

for val in grouping:
    for col in data.columns:
        if col.startswith(val):
            unknown_dict[col] = val

del unknown_dict['risk_segment_prior_ntm']
del unknown_dict['risk_segment_during_rx']

combined = [x for x in unknown_dict.keys()]

le = LabelEncoder()
for val in combined:
    data[val] = le.fit_transform(data[val])
    
test = data.set_index('ptid').groupby(unknown_dict,axis=1).sum()

test.columns = ['concomitancy_count','comorbidity_count','risk_factors_count']


test = test.reset_index()

data = data.drop(combined,axis=1)

data = pd.merge(data, test, on='ptid', how = 'inner')

data.pop('count_of_risks')

data['persistency_flag'].value_counts()



data.describe(include=['O'])

#removing useless columns

