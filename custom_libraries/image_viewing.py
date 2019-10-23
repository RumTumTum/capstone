

from custom_libraries import import_data

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

# multithreading and showing progress
import concurrent.futures
from tqdm import tqdm

# image processing
import pydicom
import cv2

# saving to cloud storage
from google.cloud import storage
from custom_libraries import gcloud_storage


def get_image(img_path, dim = (512,512)):
    """Output three versions of image - raw, windowed and resized"""
    ds=pydicom.dcmread(img_path)
    raw=ds.pixel_array
    window_center , window_width, intercept, slope = get_windowing(ds)
    windowed = window_image(raw, 50, 100, intercept, slope)
    resized = cv2.resize(windowed, dim)
    X.append(resized)
    return raw,windowed,resized

def show_images(image_paths,ncol=5):
    """
    Keyword Arguments
    image_paths - iterable of image paths
    ncol - number of columns of pictures
    """
    nrow = math.ceil(len(image_paths) / ncol)
    fig = plt.figure(figsize=(15,5*nrow))
    for i,image_path in enumerate(image_paths):
        img_list = list( get_image(image_paths[row]) )

        for sub_img in range(0,3):
            fig.add_subplot(nrow,ncol,row*3+sub_img+1)
            if bone_color:
                plt.imshow(img_list[sub_img],cmap=plt.cm.bone,**kwargs)
            else:
                plt.imshow(img_list[sub_img],**kwargs)
    #         fig.add_subplot()
    return image_paths

def display_image_processing(
    image_paths = False, 
    n_images = 5, 
    random_state=12345,
    bone_color = True,
    **kwargs):
    """Display the versions of each image
    Version:
    1. raw
    2. windowed
    3. windowed and reduced

    Keyword arguments:
    image_paths (list of strings) - image paths 
    n_images (int) - number of different images to display
    random_state (int) - random state for image selection
    
    Output:
    display (inline image) - images displayed inline
    image_paths (list) - list of image paths
    bone_color (bool) - if true, pass cmap=plt.cm.bone to plt.imshow()
    """
    if not image_paths:
        image_paths = list(image_list.origin.sample(
            n=n_images,
            replace=False,
            random_state=12345))
        # print(image_paths)
    nrow = len(image_paths)
    ncol = 3

    fig = plt.figure(figsize=(15,5*nrow))
    for row in range(0,nrow):
        img_list = list( get_image(image_paths[row]) )

        for sub_img in range(0,3):
            fig.add_subplot(nrow,ncol,row*3+sub_img+1)
            if bone_color:
                plt.imshow(img_list[sub_img],cmap=plt.cm.bone,**kwargs)
            else:
                plt.imshow(img_list[sub_img],**kwargs)
    #         fig.add_subplot()
    return image_paths
