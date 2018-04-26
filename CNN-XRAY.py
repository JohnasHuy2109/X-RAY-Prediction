# Load some basic libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import skimage
from skimage import data, transform
import random
import tensorflow as tf
tf.reset_default_graph()

workpath = 'D:\\Spyder-Workspace\\Graduattion-Project\\Images-Decompressed\\'
raw_train_dir = os.path.join(workpath, 'Raw\\Training\\')
raw_test_dir = os.path.join(workpath, 'Raw\\Testing\\')
resized_train_dir = os.path.join(workpath, 'Resized-256\\Training\\')
resized_test_dir = os.path.join(workpath, 'Resized-256\\Testing\\')
    
######################################### THIS BLOCK OF CODES ONLY RUN AT ONCE ####################################
import PIL
from PIL import Image
# IMAGE PRE-PROCESSING:
# 1. For traing images
for f in os.listdir(raw_train_dir):                                                                               
    if f.endswith(".png"):                                                                                        
        name = os.path.splitext(f)[0]                                                                             
        img = Image.open(raw_train_dir + f)
        img = img.resize((256,256), Image.ANTIALIAS)
        img.save(resized_train_dir + name + '.png')
    
# 2. For testing images
for f2 in os.listdir(raw_test_dir):
    if f2.endswith(".png"):
        name2 = os.path.splitext(f2)[0]
        img2 = Image.open(raw_test_dir + f2)
        img2 = img2.resize((256,256), Image.ANTIALIAS)
        img2.save(resized_test_dir + name2 + '.png')
###################################################################################################################

def load_data(data_directory):
    images = []
    labels = []
    with open (data_directory + 'Data_Entry_2017-fixed.csv') as file:
        csvReader = csv.reader(file)
        firstline = True;
        for row in csvReader:
            if firstline:
                firstline = False
                continue
            labels.append(row[1])
    
    file_names = [os.path.join(data_directory, d) for d in os.listdir(data_directory) if d.endswith(".png")]
    for f in file_names:
        images.append(skimage.data.imread(f, as_grey=True))
    return images, labels

images, labels = load_data(resized_train_dir)

# Processing some error items
index = 0
count = 0
new_images = []
for i in range(len(images)):
    if images[i].shape != (256, 256, 4):
        count +=1
        new_images.append(images[i])
    else:
        index = i
        del labels[index]
print("number of errors removed: ", len(images)-count)


##################################### ANALYZE DATASET ####################################################

print(np.ndim(images))
print(np.size(images))
print(np.ndim(labels)) # Dimension of 'labels'
print(np.size(labels)) # Size of 'labels'
print(len(set(labels))) # Length of 'labels'

# 1. Random Images
random_sample = [200, 2550, 3750, 4100] # Determine the (random) indexes of the images that you want to see 
# Fill out the subplots with the random images that you defined 
for i in range(len(random_sample)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[random_sample[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[random_sample[i]].shape, 
                                                  images[random_sample[i]].min(), 
                                                  images[random_sample[i]].max()))
    
# 2. All Labels
unique_labels = set(labels) # Get the unique labels 
plt.figure(figsize=(15, 15)) # Initialize the figure
i = 1 # Set a counter
# For each unique label,
for label in unique_labels:
    image = images[labels.index(label)] # pick the first image for each label 
    plt.subplot(8, 8, i) # Define 64 subplots
    plt.axis('off') # Don't include axes
    plt.title("Label {0} ({1})".format(label, labels.count(label))) # Add a title to each subplot 
    i += 1 # Add 1 to the counter
    plt.imshow(image) # plot this first image 
plt.show()