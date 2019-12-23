"""
load_from_checkpoint
    - checkpoint_path (checkpoint.pth)

process_image
    - img
    
imshow
    - Image
    - ax (None)
    - Title (None)
    

"""

import argparse
from trainfunctions import Network
from trainfunctions import data_loaders
from trainfunctions import validation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json

from PIL import Image



def load_from_checkpoint(checkpoint_path='checkpoint.pth'):
    model = models.densenet121(pretrained = True)
    
    classifier = Network(1024, 102, [516, 256], drop_p=0.5)
    model.classifier = classifier
    
    state_dict = torch.load(checkpoint_path,map_location='cpu')
    
    model.load_state_dict(state_dict)
    
    return model


    
def process_image(img):
    """
    1. Get the size of the image
    2. Get the width to height ratio in order to find what the max side length should be to maintain aspect and have
        minimum side of 256 pixels
    3. Find the cropping points to crop out the 224x224 centre portion of image
    4. Crop the image using the max value of the newly found with and height
        - this is because .thumbnail uses only the largest value and maintains the aspect based on that
    5. Crop the image based on calculated values
    6. Convert image to numpy array then normalise and transpose
        - Transpose in order to make the colour values in the correct dimension
    7. Convert to pytorch tensor and return this
    
    """
    img = Image.open(img)
    size = img.size
    width = size[0]
    height = size[1]
    ratio = width / height
    if height>width:
        new_width = 256
        new_height = 256 * (1 / ratio)
    elif height<width:
        new_width = 256 * ratio
        new_height = 256
    elif height==width:
        new_width = 256
        new_height = 256
        
    
    left = round((new_width - 224)/2)
    top = round((new_height - 224)/2)
    right = round((new_width + 224)/2)
    bottom = round((new_height + 224)/2)
    
    
    img.thumbnail((max(new_width,new_height),max(new_width,new_height)))
    #print(img.size)
    img = img.crop((left,top,right,bottom))
    np_image = np.array(img)
    
    np_image = np_image / 255
    #print(np_image.shape, 'gewonisaid')
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    #print(np_image.shape, 'should be 3 at start?')
    #print(type(torch.from_numpy(np_image)))
    
    return torch.from_numpy(np_image).float().unsqueeze(0)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
