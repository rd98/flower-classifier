
"""
Arguments
    - data_directory (REQUIRED) USE : -dir 
    - epochs (Default = 10) USE : -epochs
    - learning rate (Default = 0.001) USE : -lr

create_model
    - Takes no Arguments
    - Returns model

train_model
    - Arguments
        o model
        o trainloader
        o learning_rate (default = 0.001)
        o num_epochs (default = 10)
    
"""

import argparse
from trainfunctions import Network
from trainfunctions import data_loaders
from trainfunctions import validation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets





parser = argparse.ArgumentParser(description = "Train Image Classifier")
parser.add_argument('-dir', '--data_directory', required=True, help = 'The directory where the images are stored (default = flowers)')
parser.add_argument('-arch', '--architecture', default='dn121', help = 'Model Architecture, either dn121 or vgg19 (default=dn121)')
parser.add_argument('-GPU', '--GPU', default='T', help = 'Train on GPU if possible enter T or F (default=True)')
parser.add_argument('-epochs', '--epochs', type=int, default=10, help = 'Number of Epochs (default = 10)')
parser.add_argument('-hid', '--hidden', type=int, default=2, help = 'Number of layers in the classifier between input and output (default = 2)')
parser.add_argument('-lr', '--learningrate', type=float, default=0.001, help = 'Learning rate (default = 0.001)')

args = parser.parse_args()

def create_model(arch):
    """Returns Model"""
    if args.architecture=='dn121':
        #Load in the pretrained densenet121 model
        model = models.densenet121(pretrained = True)

        #Freeze the parameters of the pretrained model
        for param in model.parameters():
            param.requires_grad = False

        """
        Create and add a classifier to the pretrained model
            - Use the network function which is defined in the functions.py import
                o It follows the form Network(Input, Output, [Hidden layer sizes], dropout prob)    
        """
        classifier = Network(1024, 102, [516, 256], drop_p = 0.5)
        model.classifier = classifier
    
    if args.architecture=='vgg19':
        #Load in the pretrained densenet121 model
        model = models.vgg19(pretrained = True)

        #Freeze the parameters of the pretrained model
        for param in model.parameters():
            param.requires_grad = False

        """
        Create and add a classifier to the pretrained model
            - Use the network function which is defined in the functions.py import
                o It follows the form Network(Input, Output, [Hidden layer sizes], dropout prob)    
        """
        classifier = Network(25088, 102, [4096, 1000], drop_p=0.5)
        model.classifier = classifier

    return model


def train_model(model, trainloader, learning_rate, num_epochs, GPU):
    #If the GPU is available transfer model to cuda (GPU) to train faster
    if GPU=='F':
        device = 'cpu'
        print("Training on CPU")
    else:
        if torch.cuda.is_available():
            device = 'cuda'
            print("Training with GPU")
        else:
            print("GPU not available therefore training on CPU")
            device = 'cpu'
    model.to(device)
    # define criterion and optimiser
    criterion = nn.NLLLoss()
    optimiser = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    epochs = num_epochs
    print_every = 50
    steps = 0

    for e in range(epochs):
        running_loss = 0

        for images, labels in iter(trainloader):
            images, labels = images.to(device), labels.to(device)

            steps += 1
            optimiser.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()

if args.data_directory:
    trainloader, testloader, validloader = data_loaders(args.data_directory)

    model = create_model(args.architecture)
    train_model(model, trainloader, args.learningrate, args.epochs, args.GPU)
    
    torch.save(model.state_dict(), 'checkpoint.pth')



if __name__ == '__main__':
    print("this has just been run as main")
    
    