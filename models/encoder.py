import torch.nn as nn
from torchvision import models
import config

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # resnet34 = models.resnet34(pretrained=True)
        for param in resnet50.parameters():
            param.requires_grad = False
        cnn_modules = list(resnet50.children())[:-1]
        self.resnet = nn.Sequential(*cnn_modules)
        self.fc1 = nn.Linear(resnet50.fc.in_features, embed_size)
        self.batchnorm = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.fc1.weight.data.normal_(0., 0.02)
        self.fc1.bias.data.fill_(0)


    def forward(self,images):
        """
        The function takes in an image, passes it through the resnet model, flattens the output, and passes
        it through a fully connected layer.
        
        :param images: the input images
        :return: The features of the image.
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batchnorm(self.fc1(features))
        return features


