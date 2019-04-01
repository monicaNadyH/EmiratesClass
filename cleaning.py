import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()

P_THRESH = 100

def clean(text):
    t = str(text).replace('=', '').replace('#', '').replace('*', ' ').replace('-', ' ').replace('+', '').replace('_', '').replace('<', '').replace('>', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('/', '').replace('|', '').replace('.', '').replace('$', '').replace('"', '')
    " ".join(t.split())
    return t.strip()



data = pd.read_csv('EK all data sep.csv')
temp = []

data['description'] = data.apply(lambda row: clean(row['description']), axis=1) 

data['len'] = data.apply(lambda row: len(row['description']), axis=1) 
data = data[data.len < 1000]
data = data[data.len > 5]
data = data.drop(columns='len')
data.dropna(how='any')

for d in tqdm(data['description']):
    temp.append(clean(d))


data['description'] = temp
data = data[['description', 'primary', 'subcategory']]
data.to_csv('data_cleaned.csv', index=False)

primary = data[['description', 'primary']]

uniques = data['primary'].unique()
for u in tqdm(uniques):
    temp = data.query('primary == @u')
    if len(temp > P_THRESH):
        secondary = temp[['description', 'subcategory']]
        secondary.to_csv('secondary/' + u.replace('/', ' ').strip('"') + '.csv', index=False)