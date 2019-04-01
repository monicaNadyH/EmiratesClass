import pandas as pd
from tqdm import tqdm


data = pd.read_csv('data_cleaned.csv')
data.drop(columns=['description'], inplace=True)
data.dropna()

uniques = data['primary'].unique()


data.shape
data.isnull()
print(data.isnull().sum())
data['subcategory'].isnull().sum()
data=data.fillna(" NO SUB CLASS ")


res = []
for u in uniques:
    temp = data.query('primary == @u')
    t_len = len(temp)
    uni = temp['subcategory'].unique()
    l = len(uni)

    for un in uni:
        temp = data.query('subcategory == @un & primary == @u')
        s_len = len(temp)
        res.append({
            'name': u,
            'count': l,
            'total': t_len, 
            'sub': un,
            'sub_count':s_len
        })


counts = pd.DataFrame(data=res)
counts.set_index('name', inplace=True)
counts.to_csv('count_all.csv')









 
