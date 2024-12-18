import torch
from torchvision import models, transforms
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import copy
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# We use the PyTorch module made by Alexis Jacq for Neural Style Transfer components
class ContentLoss(nn.Module):
    """
    Computes the content loss between the input and the target feature maps.

    Attributes:
    ------------
    target : torch.Tensor
        The target feature map (detached to avoid gradient computation).

    Methods:
    ---------
    forward(input):
        Computes the content loss using the input feature map.
        Returns the input for compatibility with sequential models.
    """

    def __init__(self, target):
        """
        Initializes the ContentLoss module.

        Parameters:
        ------------
        target : torch.Tensor
            The target feature map from a specific layer of the model.
        """
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # Detach target to avoid affecting gradients

    def forward(self, input):
        """
        Computes the content loss between the input and the target feature map.

        Parameters:
        ------------
        input : torch.Tensor
            The input feature map from the same layer.

        Returns:
        ------------
        torch.Tensor
            The unchanged input feature map (needed for seamless integration in the model).
        """
        self.loss = get_content_loss(input, self.target)  # Store the loss
        return input  # Pass the input through unchanged


class StyleLoss(nn.Module):
    """
    Computes the style loss between the input and the target feature maps.

    Attributes:
    ------------
    target : torch.Tensor
        The target feature map (detached to avoid gradient computation).

    Methods:
    ---------
    forward(input):
        Computes the style loss using the input feature map.
        Returns the input for compatibility with sequential models.
    """

    def __init__(self, target_feature):
        """
        Initializes the StyleLoss module.

        Parameters:
        ------------
        target_feature : torch.Tensor
            The target feature map used for style loss computation.
        """
        super(StyleLoss, self).__init__()
        self.target = target_feature.detach()  # Detach target to avoid affecting gradients

    def forward(self, input):
        """
        Computes the style loss between the input and the target feature map.

        Parameters:
        ------------
        input : torch.Tensor
            The input feature map from the same layer.

        Returns:
        ------------
        torch.Tensor
            The unchanged input feature map (needed for seamless integration in the model).
        """
        self.loss = get_style_loss(input, self.target)  # Store the loss
        return input  # Pass the input through unchanged


class Normalization(nn.Module):
    """
    Normalizes an image using the mean and standard deviation of the dataset.

    Attributes:
    ------------
    mean : torch.Tensor
        The mean values for each channel (reshaped to match image dimensions).
    std : torch.Tensor
        The standard deviation values for each channel (reshaped to match image dimensions).

    Methods:
    ---------
    forward(img):
        Normalizes the input image tensor using the mean and standard deviation.
    """

    def __init__(self, mean, std):
        """
        Initializes the Normalization module.

        Parameters:
        ------------
        mean : list or torch.Tensor
            The mean values for each channel (e.g., for ImageNet: [0.485, 0.456, 0.406]).
        std : list or torch.Tensor
            The standard deviation values for each channel (e.g., for ImageNet: [0.229, 0.224, 0.225]).
        """
        super(Normalization, self).__init__()
        # Convert mean and std to tensors and reshape for broadcasting across the image dimensions
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        """
        Normalizes the input image.

        Parameters:
        ------------
        img : torch.Tensor
            The input image tensor to be normalized.

        Returns:
        ------------
        torch.Tensor
            The normalized image tensor.
        """
        return (img - self.mean) / self.std  # Apply normalization
    
def image_loader(image_name,imshape):
        # scale imported image
        # transform it into a torch tensor
        loader = transforms.Compose([transforms.Resize(imshape),  transforms.ToTensor()])

        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)   # add an additional dimension for fake batch (here 1)
        return image.to(device, torch.float) # move the image tensor to the correct device



def style_transfer(content_image_name, style_image_name, style_weight, content_weight, imshape, num_steps):

    #Load the images
    image_path = "../data/"
    content_image = image_loader(image_path + content_image_name, imshape)
    style_image = image_loader(image_path + style_image_name, imshape)

    # Initialize the target image
    input_image = torch.randn(content_image.data.size(), device=device)
    input_image.requires_grad_(True)

    # Load the pretrained model
    model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)

    model = model.features.eval() # set the model to evaluation mode

    model_normalization_mean = [0.485, 0.456, 0.406]
    model_normalization_std = [0.229, 0.224, 0.225]
    
    # Copy our model to iterate on the layers after
    vgg = copy.deepcopy(model)

    # Create a normalization layer with specified mean and std
    normalization = Normalization(model_normalization_mean, model_normalization_std).to(device)
    # Add the layer to our model
    model = nn.Sequential(normalization)

    blocks = [2, 2, 4, 4, 4]  # Number of convolutional layers in each block of the VGG-19 model

    # Name of the layers for the style and content representation
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    content_layer = 'conv4_2'

    style_losses = []
    content_losses = []
    index_conv = 0
    current_block = 0

    i = 0  

    for layer in vgg.children():
        # Check if we need to move on to the next block
        if current_block < len(blocks) and index_conv == blocks[current_block]:
            index_conv = 0  # Réinitialiser le compteur pour le nouveau bloc
            current_block += 1

        if isinstance(layer, nn.Conv2d):  # For the convolutional layers
            index_conv += 1
            name = f'conv{current_block + 1}_{index_conv}'
            model.add_module(name, layer)  # Add the layer to the model

            # Add style and content losses
            if name in style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)

            if name == content_layer:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module(f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)

        elif isinstance(layer, nn.ReLU):  
            name = f'relu{current_block + 1}_{index_conv}'
            model.add_module(name, nn.ReLU(inplace=False))
            
        elif isinstance(layer, nn.MaxPool2d): 
            name = f'pool{current_block + 1}'
            model.add_module(name, layer)

        i += 1

    # Print the first layers to check the naming
    print(model)

    # Retrieve only the subpart of the model that we need for the style transfer
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i + 1]

    optimizer = optim.LBFGS([input_image.requires_grad_()])

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
                style_score += (1/5)*style_loss.loss

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


if __name__ == "__main__":
    style_transfer(
        content_image_name="content.jpeg",
        style_image_name="style.jpeg",
        style_weight=10**4,
        content_weight=1,
        imshape=(224,224),
        num_steps=3000
    )