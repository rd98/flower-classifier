import argparse
from trainfunctions import Network
from trainfunctions import data_loaders
from trainfunctions import validation

from predictfunctions import load_from_checkpoint, process_image, imshow

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import json



parser = argparse.ArgumentParser(description = "Train Image Classifier")
parser.add_argument('-img', '--image_path', type=str, required=True, help = 'The image path')
parser.add_argument('-cp', '--checkpoint', type=str, help = 'The checkpoint where the trained model is stored (default=checkpoint.pth)')
parser.add_argument('-topk', '--topk', type=int, default=1, help = 'The top number of classes you wish to see')
parser.add_argument('-JSON', '--JSON', default = 'cat_to_name.json', type=str, help = 'The JSON file you wish to map class values with (default=cat_to_name.json)')
parser.add_argument('-GPU', '--GPU', default='F', help = 'Train on GPU if possible enter T or F (default=False)')

args = parser.parse_args()

with open(args.JSON, 'r') as f:
    cat_to_name = json.load(f)
cat_to_name_temp = sorted(cat_to_name)

if args.checkpoint:
    model = load_from_checkpoint(args.checkpoint)
else:
    model = load_from_checkpoint()
    
def predict(image_path, model, GPU, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if GPU=='F':
        device = 'cpu'
    else:
        device = 'cuda'
    
    model.to(device)
    image = image_path.to(device)
    
    model.eval()
    output = model.forward(image)
    ps = torch.exp(output)
    
    results = torch.topk(ps, topk)
    prob_values = []
    for i in range(len(results[0][0])):
        prob_values.append(float(torch.topk(ps,args.topk)[0][0][i].cpu().detach().numpy()))

    class_values = []
    for i in range(len(results[1][0])):
        class_values.append(float(torch.topk(ps,args.topk
                                            )[1][0][i].cpu().detach().numpy()))

    #print(prob_values)
    #print(class_values)

    classes=[]
    #print(cat_to_name_temp)
    for i in class_values:
        classes.append(cat_to_name[str(cat_to_name_temp[int(i)])])

    #print(classes)
    
    for prob, pred in zip(prob_values, classes):
        print("Probability: {}        Class: {}".format(prob, pred))


trainloader, testloader, validloader = data_loaders('flowers')

# if args.image_path:
#     predict(process_image(args.image_path), model, args.topk)
# else:
#     count=0
#     for image, label in validloader:
#         predict(image, model, args.topk)
#         count+=1
#         if count>0:
#             break

if __name__=='__main__':
    predict(process_image(args.image_path), model, args.GPU, args.topk)



        
        