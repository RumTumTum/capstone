# Standard libraries
import numpy as np
import pandas as pd

# Keras and Tensorflow
import tensorflow as tf
import keras

# saving to cloud storage
from datetime import datetime
from google.cloud import storage
from custom_libraries import gcloud_storage #local py file


def save_model(model, model_tags = [],return_df=False, model_notes = ""):
    """
    Save model in the designated root folder
    Add a row to the model_records.csv with information about the current training run
    Keyword arguments:
    model -- keras model
    """
    
    # get the current time
    d = datetime.utcnow()
    timestamp = d
    time_name = d.strftime("%Y-%m-%dT%H%M%S")

    # set folder paths and other saving specs
    root = 'models/'
    save_as_type = 'h5' # can choose 'h5' or 'tf'
    model_records_path = root+'model_records.csv'
    model_name = time_name+'_model'+'.'+save_as_type
    model_path = root+model_name
    
    
    # model attributes
    train_size = 1
    validation_size = 1    
    #     train_data_class_imbalance = 
    epochs = 1
    optimizer = 1
    layers = []
    num_layers = len(layers)

    # create a unique set of tags, include tensorflow and keras
    model_tags = set([x.lower() for x in model_tags + ['tensorflow','keras']])

    # model effectiveness    
    score = 1
    
    # training information
    total_time = 1
    
    data = {
    # general
        'timestamp': [timestamp],
        'model_notes': [model_notes],
    # model attributes
        'train_size': [train_size],
        'validation_size': [validation_size],
        'epochs': [epochs],
        'optimizer': [optimizer],
        'num_layers': [num_layers],
        'model_tags': [model_tags],
        'layers': [layers],
    # model effectiveness
#         'loss': [loss],
        # added below
        'score': [score],
    # training specification
    # saving information
        'model_name': [model_name],
        'model_path': [model_path]
    }

#     for attribute in model.metrics_names:
#         try:
#             data[attribute] = model[attribute]
#         except:
#             print("model."+attribute+" is not a valid parameter")
    
    # save the data to a dictionary
    temp_df = pd.DataFrame.from_dict(data)
    
    # attempt to amend the previous csv. If not available, create a new one
    
#     try:
#         load from google
#     except:
#         model_records_path = root + "model_records_conflicting_" + time_name + ".csv"

    try:
        df = pd.read_csv(model_records_path)
        df = pd.concat([df,temp_df],sort=False)
    except:
        df = temp_df
    finally:
        df.reset_index(drop=True,inplace=True)
    
    # save df to csv
    df.to_csv(model_records_path,index=False)
    
    # trying with save_model instead of model.save
    keras.models.save_model(model, model_path)
    
#     # save csv to cloud
#     try:
#         upload_blob("fi-capstone-data",model_path,model_path)
#         upload_blob("fi-capstone-data",model_records_path,model_records_path)
#     except:
#         print("MODEL AND RECORDS NOT SAVED TO CLOUD")
    

    if return_df:
        return df