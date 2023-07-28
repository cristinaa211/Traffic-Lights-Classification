from abc import ABC, abstractmethod
from tools.read_files import read_files_multiple_directories
from tools.image_operations import read_image_PIL, convert_to_pytorch_tensor
from torch.utils.data import Dataset


class CustomDataset(ABC):
    @abstractmethod
    def load_data(self, path):
        """Returns a list of data from a given path."""
        pass

    def label_data(self, images_path):
        """Labels the data and returns a list of data-label pairs."""
        pass

    def create_dataset(self, path):
        """Returns the final dataset as a list."""
        pass

class CustomImageDataset(CustomDataset):
    """Concrete Class"""
    def __init__(self) -> None:
        super().__init__()

    def load_data(self, images_path):
        """Returns a list of images and their paths."""
        images = []
        for single_path in images_path:
            im = read_image_PIL(single_path)
            images.append(im)
        print("There are {} images.".format(len(images)))
        return images
    
    def label_data(self, images_path):
        """Labels data depending on the folder path. If the image is in the "green" folder, it will have the "0" label."""
        labels_map = {'green' : 0, 'red' : 1}
        labels = []
        for img_path in images_path:
            if 'green' in img_path:
                labels.append(labels_map['green'])
            elif 'red' in img_path:
                labels.append(labels_map['red'])
        print("There are {} labels.".format(len(labels)))
        return labels

    def create_dataset(self, path):
        """Returns the final dataset."""
        images_path = read_files_multiple_directories(path)
        images = self.load_data(images_path)
        labels = self.label_data(images_path)
        return images, labels
    

class DataLoader(Dataset):
    def __init__(self, custom_set):
        super().__init__()
        self.data = [custom[0] for custom in custom_set]
        self.label = [custom[1] for custom in custom_set]

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        return image, label 

    def yield_data(self, index):
        yield self.data[index], self.label[index]