from torchvision import transforms
import torch


transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.functional.autocontrast()
])