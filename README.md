# Image Classifier

In this project, I trained an image classifier to recognise different species of flowers. I used this dataset of 102 flower categories. Link to download the data can be found in the Jupyter Notebook file.

This project can be easily genaralised to train on another labelled photo dataset.

In the Jupyter Notebook I demonstrate the trained model. The end of the notebook implements the model on images downloaded from Google Images. The training code can be seen in the .py files in the repository here and is implemented via a command line application.

### Further Backgroud Info

In this project I used transfer learning to harness the power of a pretrained deep neural network and added a further three hidden layers to act as a classifier. This allowed the model to have a high accuracy while drastically reducing the training required to identify features in the images.

The pre-trained network I used was trained on the ImageNet dataset where each color channel was normalized separately. Thus, I had to normalize the means and standard deviations of the images to what the network expects. In the case of the network I used, densenet121, it required a shift of each color channel to be centered at 0 and range from -1 to 1. Also, the pretrained networks take images of size 224x224 and so inputs must be resized as required.

The category-name relationships were given via a .json file and thus had to be read in with the json module. This gives a dictionary mapping the integer encoded categories to the actual names of the flowers.

