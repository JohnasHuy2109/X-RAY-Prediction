import cv2
import os
import time

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def crop_and_resize_image(path, new_path, img_size):
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    new_height = img_size
    new_weight = img_size
    for items in dirs:
        img = cv2.imread(path + items, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (new_weight, new_height))
        cv2.imwrite(str(new_path + items), img)
        
if __name__ == "__main__":
    ROOT_PATH = "E:\\FOR UNIVERSITY\\Special-Document\\Graduation-Project\\Images\\"
    Raw_path = os.path.join(ROOT_PATH, "Raw\\")
    Resized_path = os.path.join(ROOT_PATH, "Resized-128\\")
    start_time = time.time()
    crop_and_resize_image(path=Raw_path, new_path=Resized_path, img_size=128)
    print("Second: ", time.time() - start_time)
    print("Done!")
    
