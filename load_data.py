import os
import cv2
import numpy as np
from extract_features import normalize_image

class Load_data_image():
    def __init__(self,path, label_it = True):
        self.path = path
        self.label_it = label_it

    def load_images_from_path(self):
        list_image_vectors = [] # the final list with all the data
        files = os.listdir(self.path)
        for file in files :       
            image = cv2.imread(r'{}/{}'.format(self.path, file)) # reading each image in the folder path
            try:
                image = cv2.resize(image, (200,200), interpolation=cv2.INTER_AREA)
                image_norm = normalize_image(image) # Normalize the dataset
                image_norm_array = np.array(image_norm, dtype = np.float16) 
                image_flat = np.matrix.flatten(image_norm_array)
                list_image_vectors.append(image_flat) # Adding the data to the final data list
            except: pass
        print(f"There are {len(list_image_vectors)} pictures.")
        return list_image_vectors

    def load_images(self):
        list_image_vectors = self.load_images_from_path()
        if self.label_it == True:
            if 'green' in self.path : label = 1.0
            elif 'red' in self.path : label = 0.0
            labels = np.ones((len(list_image_vectors), 1)) * label
            return list_image_vectors,labels
        else: return list_image_vectors, False
        

class Process_data:
    def __init__(self, data) -> None:
        self.data = data

    def transform_data(self):
