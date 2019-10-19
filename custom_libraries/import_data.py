# Standard libraries
import numpy as np
import pandas as pd

# Shuffle library
from sklearn.utils import shuffle

def clean_data_csv(data_path = "../data/", csv_name = 'stage_1_train.csv',shuffle_data = True):

    data_raw = pd.read_csv(data_path + csv_name)
    data_raw['filename'] = data_raw['ID'].apply(lambda x: "ID_" + x.split('_')[1] + ".dcm")
    data_raw['type'] = data_raw['ID'].apply(lambda x: x.split('_')[2])

    data_pivot = data_raw[['Label', 'filename', 'type']].drop_duplicates().pivot(
        index='filename', columns='type', values='Label').reset_index()
    
    if shuffle_data:
        data = shuffle(data_pivot)
    else:
        data = data_pivot
    data.reset_index(drop=True,inplace=True)
        
    return data
