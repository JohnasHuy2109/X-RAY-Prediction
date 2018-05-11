import time

import cv2
import numpy as np
import pandas as pd

def convert_image_to_arr(file_path, df):
    lst_imgs = [l for l in df['Image Index']]
    return np.array([np.array(cv2.imread(file_path+img, cv2.IMREAD_GRAYSCALE)) for img in lst_imgs])

def save_to_arr(arr_name, arr_obj):
    return np.save(arr_name, arr_obj)

if __name__ == "__main__":
    ROOT_PATH = "E:\\FOR UNIVERSITY\\Special-Document\\Graduation-Project\\Images\\"
    start_time = time.time()
    labels = pd.read_csv(ROOT_PATH + 'Data_samples2.csv')
    
    print("Writing training array.....")
    X_train = convert_image_to_arr(ROOT_PATH + 'Resized-128\\', labels)
    print("Done!")
    print(X_train.shape)
    
    print("Saving training array.....")
    save_to_arr(ROOT_PATH + "X_train2.npy", X_train)
    print ("Done!")
    print("Seconds: ", round((time.time() - start_time), 2))
