# Standard libraries
import numpy as np
import pandas as pd

# Various
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


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
    data = pd.DataFrame(data,columns = list(data.columns))
        
    return data

def balanced_images_all_classes(image_list,n=2500,replace=False,random_state = 12345):
    image_subset = [
        image_list[image_list[category] ==1 ].sample(
            n=n,
            replace = replace, 
            random_state = random_state)
        for category
        in ['epidural',
            'intraparenchymal',
            'intraventricular',
            'subarachnoid',
            'subdural']
    ]
    image_subset.append(
        image_list[image_list['any'] == 0 ].sample(
            n=n*5,
            replace = replace, 
            random_state = random_state)
    )
   
    image_subset_combined = pd.concat(image_subset).drop_duplicates()
    image_subset_combined = shuffle(image_subset_combined,random_state = random_state)
    image_subset_combined.reset_index(drop=True, inplace = True)
    return image_subset_combined


def balanced_images_binary(image_list,n=2500,replace=False,random_state = 12345):
    image_subset = [image_list[image_list['any'] ==1 ]]
    image_subset.append(
        image_list[image_list['any'] == 0 ].sample(
            n=len(image_subset[0]),
            replace = replace, 
            random_state = random_state)
    )
    
    image_subset_combined = pd.concat(image_subset).drop_duplicates()
    image_subset_combined = shuffle(image_subset_combined,random_state = random_state)
    image_subset_combined.reset_index(drop=True,inplace=True)
    return image_subset_combined