import torch
from torchvision import models
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import copy
from ..utils.utils import image_loader

from PIL import Image

def style_transfer(content_image_name, style_image_name, style_weight, content_weight):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # shape of the output image
    imshape = (512, 512)

    image_path = "../data/"
    content_image = image_loader(image_path + content_image_name)
    style_image = image_loader(image_path + style_image_name)

    input_image = torch.randn(content_image.data.size(), device=device)
    input_image.requires_grad_(True)

    model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)

    model = model.features.eval() # set the model to evaluation mode

    model_normalization_mean = [0.485, 0.456, 0.406]
    model_normalization_std = [0.229, 0.224, 0.225]

    def get_content_loss(input, target):
        return F.mse_loss(input, target)

    def gram_matrix(input):
        a, b, c, d = input.size()  
        # a = batch size, 
        # b = number of feature maps, 
        # c, d = height, width
        features = input.view(a * b, c * d)   # flatten the feature maps
        G = torch.mm(features, features.t())  # compute the dot product of the feature maps
        return G.div(a * b * c * d)           # normalize the Gram matrix

    def get_style_loss(input, target):
        G_input = gram_matrix(input)
        G_target = gram_matrix(target)
        return F.mse_loss(G_input, G_target)
    
    class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = get_content_loss(input, self.target)
            return input

    class StyleLoss(nn.Module):

        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = target_feature.detach()

        def forward(self, input):
            self.loss = get_style_loss(input, self.target)
            return input
        
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # set mean and std as tensors, reshaped to match image dimensions
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        # Forward method to normalize input image
        def forward(self, img):
            return (img - self.mean) / self.std
    
    # Copy our model to iterate on the layers after
    vgg = copy.deepcopy(model)

    # Create a normalization layer with specified mean and std
    normalization = Normalization(model_normalization_mean, model_normalization_std).to(device)
    # Add the layer to our model
    model = nn.Sequential(normalization)

    content_losses = []
    style_losses = []

    i = 0

    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)


        if name in content_layers_default:
            # add content loss:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers_default:
            # add style loss:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i + 1]

    optimizer = optim.LBFGS([input_image.requires_grad_()])

    num_steps = 500

    recent_images = []

    step_counter = [0]
    while step_counter[0] <= num_steps:

        # Recall function for the optimiser
        def closure():
            input_image.data.clamp_(0, 1)

            optimizer.zero_grad()     # do not accumulate parameter gradients
            model(input_image)        # passes the image as input to the model
            style_score = 0
            content_score = 0

            # add up the losses for each style layer
            for style_loss in style_losses:
                style_score += style_loss.loss

            # add up the losses for each content layer
            for content_loss in content_losses:
                content_score += content_loss.loss

            # apply the weights
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score # calculate the total loss
            loss.backward()                    # backward to calculate the gradients of the input image

            step_counter[0] += 1

            if step_counter[0] % 50 == 0:      # display style and content scores every 50 steps
                print("step {}:".format(step_counter[0]))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()


            return style_score + content_score

        optimizer.step(closure)

    input_image.data.clamp_(0, 1)

    return input_image
