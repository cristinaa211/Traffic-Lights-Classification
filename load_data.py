import os
import cv2
import numpy as np


class Load_data_image():
    def __init__(self,path, label_it = True):
        self.path = path
        self.label_it = label_it

    def load_images(self):
        # the final list with all the data
        data = []
        files = os.listdir(self.path)
        for file in files :
            # reading each image in the folder path
            image = cv2.imread(r'{}/{}'.format(self.path, file))
            if image is None:
                pass
            else:
                image = cv2.resize(image, (200,200), interpolation=cv2.INTER_AREA)
                # Normalize the dataset
                image = np.array(image, dtype = np.float16) / 255.0
                image = np.matrix.flatten(image)
                # Adding the data to the final data list
                data.append(image)
        return data

    def forward(self):
        data = self.load_images()
        if self.label_it == True:
            if 'green' in self.path :
                label = 1.0
            elif 'red' in self.path :
                label = 0.0
            labels = np.ones((len(data), 1)) * label
            return data,labels
        else:
            return data, False