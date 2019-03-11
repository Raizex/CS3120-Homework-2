#Credit to Tovio Roberts for writing most of the Image IO code, which can be found here:
#https://github.com/clownfragment/CS3120_ML_suppl_mats

import cv2
from os import listdir, getcwd
from skimage.io import imread
import matplotlib.pyplot as plt

def plot_image(img,title):
    #Plot image
    fig, ax1 = plt.subplots(ncols=1, figsize=(18, 6), sharex=True,
                            sharey=True)
    ax1.imshow(img, cmap='gray')
    ax1.set_title(title)
    ax1.axis('off')

subfolder = 'data/KNN/animals/'
classes = listdir(subfolder)
classes.sort()

cats_filenames = listdir(subfolder + classes[0])
cats_filenames.sort()
cats_filenames = [subfolder + classes[0] + '/' + \
                  filename for filename in cats_filenames]
cat0_rgb = imread(cats_filenames[0])
plot_image(cat0_rgb, cats_filenames[0] + 'original')

cat0_resize = cv2.resize(cat0_rgb, (32, 32),interpolation=cv2.INTER_CUBIC)
plot_image(cat0_resize, cats_filenames[0] + 'resized')

cat0_vect = cat0_resize.reshape(1,3072)
print(cat0_vect.shape)