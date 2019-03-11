#Credit to Tovio Roberts for writing most of the Image IO code, which can be found here:
#https://github.com/clownfragment/CS3120_ML_suppl_mats

import cv2
import pandas as pd
import numpy as np
from os import listdir, getcwd
from skimage.io import imread
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', -1)

def plot_image(img,title):
    #Plot image
    fig, ax1 = plt.subplots(ncols=1, figsize=(18, 6), sharex=True,
                            sharey=True)
    ax1.imshow(img, cmap='gray')
    ax1.set_title(title)
    ax1.axis('off')

#Credit to kztd for generating dataframe from lists: https://stackoverflow.com/a/42723801    
cols = ['filename','label','image']
lst = []

subfolder = 'data/KNN/animals/'
classes = listdir(subfolder)
classes.sort()

for clas in classes:
    class_filenames = listdir(subfolder + clas)
    class_filenames.sort()
    class_filenames = [subfolder + clas + '/' + \
                       filename for filename in class_filenames]
    
    for animal_filename in class_filenames:
        print('Processing: ' + animal_filename + '\n')
        animal_rgb = imread(animal_filename)
        animal_resize = cv2.resize(animal_rgb, (32, 32),interpolation=cv2.INTER_CUBIC)
        
        try:
            animal_vect = animal_resize.reshape(3072)
        except ValueError:
            color_img = cv2.cvtColor(animal_resize, cv2.COLOR_GRAY2RGB)
            plot_image(color_img,animal_filename + 'resize_and_convert')
            animal_vect = color_img.reshape(3072)
            
        lst.append([animal_filename, clas[:-1], animal_vect])
        
array = np.array(lst)

data = pd.DataFrame(array, columns=cols)       
data.to_pickle('data.pk')