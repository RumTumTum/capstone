# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Images
import pydicom
import cv2

# Batching for DataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence

# Keras and Tensorflow
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs_labels, data_path = '', batch_size=128, dim=(512,512)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs_labels.reset_index(drop=True)
        self.data_path = data_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs['filename'][k] for k in indexes]
        list_label_temp=[
            [int(self.list_IDs['any'][i]),
             int(self.list_IDs['epidural'][i]),
             int(self.list_IDs['intraparenchymal'][i]),
             int(self.list_IDs['intraventricular'][i]),
             int(self.list_IDs['subarachnoid'][i]),
             int(self.list_IDs['subdural'][i])] 
            for i in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp,list_label_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp,list_label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            ds=pydicom.dcmread(self.data_path+'stage_1_train_images/' +list_IDs_temp[i] )
            raw=ds.pixel_array
            window_center , window_width, intercept, slope = get_windowing(ds)
            windowed = window_image(raw, 50, 100, intercept, slope)
            resized = cv2.resize(windowed, self.dim)
            X.append(resized)
        X=np.array(X).reshape(-1,self.dim[0],self.dim[1],1) # remove 1 or reshape to get viewable image
        y_train=np.asarray(list_label_temp) 
        return X,y_train
    
class DataGenerator_single_class(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs_labels, data_path = '', batch_size=128, dim=(512,512)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs_labels.reset_index(drop=True)
        self.data_path = data_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs['filename'][k] for k in indexes]
        list_label_temp=[
            [int(self.list_IDs['any'][i])]
            for i in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp,list_label_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp,list_label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            ds=pydicom.dcmread(self.data_path+'stage_1_train_images/' +list_IDs_temp[i] )
            raw=ds.pixel_array
            window_center , window_width, intercept, slope = get_windowing(ds)
            windowed = window_image(raw, 50, 100, intercept, slope)
            resized = cv2.resize(windowed, self.dim)
            X.append(resized)
        X=np.array(X).reshape(-1,self.dim[0],self.dim[1],1) # remove 1 or reshape to get viewable image
        y_train=np.asarray(list_label_temp) 
        return X,y_train    