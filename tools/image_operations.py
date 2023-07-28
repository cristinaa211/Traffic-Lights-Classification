from PIL import Image 
import torchvision
from torchvision import transforms


def read_image_PIL(single_path):
    """Reads an image path and returns a PIL image object."""
    return Image.open(r'{}'.format(single_path))

def convert_to_pytorch_tensor(image):
    """Converts an image to a PyTorch tensor with values in the range [0,1]"""
    return transforms.ToTensor()(image)

def read_image_torch(single_path):
    """Read a JPEG or PNG image to a 3D RGB or grayscale Tensor"""
    return torchvision.io.read_image(single_path)


