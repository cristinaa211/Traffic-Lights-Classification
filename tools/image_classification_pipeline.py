from abc import ABC, abstractmethod
from tools.image_operations import  display_random_images
from tools.custom_dataset import CustomImageDataset

class ImageProcessingPipeline(ABC):
    @abstractmethod
    def load_images(self, path):
        """Abstract method to load an image from the given file path."""
        pass
    @abstractmethod
    def apply_transformations(self, images):
        """Abstract method to apply image transformations."""
        pass

class ImageProcessing(ImageProcessingPipeline):
    """Concrete Class"""
    def __init__(self):
        super().__init__()

    def load_images(self, path):
        """Load images from a given path.
        It returns two lists, tensors and labels."""
        custom_set = CustomImageDataset()
        tensors, labels = custom_set.create_dataset(path)  
        return tensors, labels 

    def apply_transformations(self, images):
        """Apply Transformations on the images."""
        return images
    
    def transform_images(self, images):
        return self.apply_transformations(images)

    def processed_images(self, path, display = True):
        labels_map = {0 : 'green', 1: 'red'}
        tensors, labels = self.load_images(path)
        transformed_imgs = self.transform_images(tensors)
        if display == True:
            display_random_images(transformed_imgs, labels, labels_map)
        return transformed_imgs, labels 
        
class ImageFeaturesExtraction(ABC):
    @abstractmethod
    def extract_features(self, image):
        """Abstract method to extract features from the image."""
        pass

    @abstractmethod
    def store_features(self):
        """Abctract method to store extracted features."""
        pass
