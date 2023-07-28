from abc import ABC, abstractmethod
from tools.data_exploration import  display_random_images
from tools.custom_dataset import CustomImageDataset
from torchvision import transforms

class ImageProcessingPipeline(ABC):
    @abstractmethod
    def load_images(self, path):
        """Abstract method to load images from the given file path."""
        pass
    @abstractmethod
    def apply_transformations(self, images):
        """Abstract method to apply image transformations."""
        pass
    @abstractmethod
    def transform_images(self, images):
        """Return the set of transformed images"""
        pass
    @abstractmethod
    def process_images(self, path, labels_map, display):
        """Given a path, it load the images from multiple directories, apply transforms operations 
        and returns a list of tensors and their corresponding labels
        based on the labels_map dictionary. Transformations are applied on images.
        If display is set to True, it will display random images from the dataset.  """


class ImageProcessing(ImageProcessingPipeline):
    """Concrete Class"""
    def __init__(self):
        super().__init__()

    def load_images(self, path):
        """Load images from a given path.
        It returns two lists, PIL images and labels."""
        custom_set = CustomImageDataset()
        images, labels = custom_set.create_dataset(path)  
        return images, labels 

    def apply_transformations(self, images):
        """Apply Transformations on the images."""
        transformations = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomAutocontrast(0.55),
                        transforms.ToTensor()
                        ])
        trans_images = []
        for image in images:
            trans_images.append(transformations(image))
        return trans_images
    
    def transform_images(self, images):
        return self.apply_transformations(images)

    def process_images(self, path, labels_map, display = True):
        images, labels = self.load_images(path)
        transformed_imgs = self.transform_images(images)
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
