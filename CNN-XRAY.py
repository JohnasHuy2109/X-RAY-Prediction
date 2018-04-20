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
raw_train_dir = os.path.join(workpath, 'Training\\')
raw_test_dir = os.path.join(workpath, 'Testing\\')
temp = os.path.join(workpath, 'Test.temp\\')
resized_train_dir = os.path.join(workpath, 'Training512\\')
resized_test_dir = os.path.join(workpath, 'Testing512\\')
    
######################################### THIS BLOCK OF CODES ONLY RUN AT ONCE ####################################
import PIL
from PIL import Image
# IMAGE PRE-PROCESSING:
# 1. For traing images
for f in os.listdir(raw_train_dir):                                                                               
    if f.endswith(".png"):                                                                                        
        name = os.path.splitext(f)[0]                                                                             
        img = Image.open(raw_train_dir + f)
        img = img.resize((512,512), Image.ANTIALIAS)
        img.save('D:\\Spyder-Workspace\\Graduattion-Project\\Images-Decompressed\\Training512\\' + name + '.png')
    
# 2. For testing images
for f2 in os.listdir(raw_test_dir):
    if f2.endswith(".png"):
        name2 = os.path.splitext(f2)[0]
        img2 = Image.open(raw_test_dir + f2)
        img2 = img2.resize((512,512), Image.ANTIALIAS)
        img.save('D:\\Spyder-Workspace\\Graduattion-Project\\Images-Decompressed\\Testing512\\' + name2 + '.png')
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
    if images[i].shape != (32, 32, 4):
        count +=1
        new_images.append(images[i])
    else:
        index = i
        del labels[index]
print("number of errors removed: ", len(images)-count)

print(np.ndim(new_images))
print(np.size(new_images))

random_sample = [200, 2550, 3750, 4100] # Determine the (random) indexes of the images that you want to see 
# Fill out the subplots with the random images that you defined 
for i in range(len(random_sample)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(new_images[random_sample[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[random_sample[i]].shape, 
                                                  images[random_sample[i]].min(), 
                                                  images[random_sample[i]].max()))