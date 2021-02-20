import torch
#import torchvision
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
import torch.utils.data
#import torch.optim as optim
#import torch.nn as nn
import os
#import time
import numpy as np
from PIL import Image


dnn = 'alexnet'
model = torch.hub.load('pytorch/vision:v0.6.0', dnn, pretrained=True)


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

data_path = 'image_net/'

entries = os.listdir(data_path)

model_ft = model
model_ft.eval()

for entry in entries:
    picture = data_path + entry
    print( np.argmax(model_ft(image_loader(data_transforms,picture)).detach().numpy()))
