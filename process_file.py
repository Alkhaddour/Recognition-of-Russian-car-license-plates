import os

import pandas as pd

filepath = './dataset/classification/aggregated_results_by_ds__pool_28674905__2021_10_16.tsv'

data = pd.read_csv(filepath, sep='\t')
cols = {'INPUT:img_path': 'img_path',
        'OUTPUT:label': 'label'}
data = data[cols.keys()]
data.columns = list(cols.values())


def get_name(path):
    return os.path.join('./dataset/classification/train_unlabelled/',
           os.path.split(path)[1])


def change_label(lbl):
    if lbl == -1:
        lbl = 0
    return lbl


data['img_path'] = data['img_path'].apply(get_name)
print(data['label'].unique())
print(data.groupby(['label']).count())
data['label'] = data['label'].apply(change_label)
print(data.groupby(['label']).count())
print(data['label'].unique())

data.to_csv('./dataset/classification/train_unlabelled.csv', index=False)
