from PIL import Image 
import torchvision
from torchvision import transforms
import random
import matplotlib.pyplot as plt

def read_image_PIL(single_path):
    """Reads an image path and returns a PIL image object."""
    return Image.open(r'{}'.format(single_path))

def convert_to_pytorch_tensor(image):
    """Converts an image to a PyTorch tensor with values in the range [0,1]"""
    return transforms.ToTensor()(image)

def read_image_torch(single_path):
    """Read a JPEG or PNG image to a 3D RGB or grayscale Tensor"""
    return torchvision.io.read_image(single_path)

def transforms_set():
    transform_operations = transforms.Compose([
        transforms.resize(256, interpolation = 3)
        
    ])


def display_random_images(images, labels, labels_map):
    random_indexs = random.sample(range(0, len(images)), 9)
    fig = plt.figure(figsize=(16, 10))
    for idx in random_indexs:
        subplot_index = random_indexs.index(idx)
        image = transforms.ToPILImage()(images[idx])
        plt.subplot(3, 3, subplot_index+1)
        plt.imshow(image)
        plt.title(labels_map[labels[idx]])
    plt.show()