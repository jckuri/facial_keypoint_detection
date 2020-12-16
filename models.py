## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        # output_size = (W-F)/S+1
        WIDTH = 224
        n_outputs = 68 * 2
        basic_colors = 1
        n_features1 = 20
        n_features2 = 30
        n_features3 = 40
        n_features4 = 50
        kernel = 3
        self.conv1 = nn.Conv2d(basic_colors, n_features1, kernel) 
        output_size1 = int(0.5 * ((WIDTH - kernel) / 1 + 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_features1, n_features2, kernel) 
        output_size2 = int(0.5 * ((output_size1 - kernel) / 1 + 1))
        self.conv3 = nn.Conv2d(n_features2, n_features3, kernel) 
        output_size3 = int(0.5 * ((output_size2 - kernel) / 1 + 1))
        self.conv4 = nn.Conv2d(n_features3, n_features4, kernel) 
        output_size4 = int(0.5 * ((output_size3 - kernel) / 1 + 1))
        linear1 = n_features4 * output_size4 * output_size4
        linear2 = int(0.5 * linear1) #4000 #1000
        linear3 = 1000
        print('output_sizes={}'.format([output_size2, output_size3, output_size4]))
        print('linear={}'.format([linear1, linear2, linear3, n_outputs]))
        self.fc1 = nn.Linear(linear1, linear2)
        self.drop = nn.Dropout(p = 0.25) 
        self.fc2 = nn.Linear(linear2, linear3)
        self.fc3 = nn.Linear(linear3, n_outputs)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
