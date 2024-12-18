import torch
import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
import torch.nn as nn
import os
from PIL import Image
import matplotlib.pyplot as plt

# use cuda if it is available for GPU training otherwise it will use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imshape = (224, 224)

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

class VGGActivationsStyle(nn.Module):
    """
    Extracts activations from specific layers of a VGG model and computes their Gram matrices 
    for style transfer tasks, returning a structured dictionary.
    """
    def __init__(self, model, target_layers):
        """
        Initializes the class with the given model and the list of target layers.
        """
        super(VGGActivationsStyle, self).__init__()
        self.model = model
        self.target_layers = target_layers
        self.layer_outputs = {}

    def gram_matrix(self, activation):
        """
        Computes the Gram matrix of the activation to capture style information.
        """
        a, b, c, d = activation.size()  
        features = activation.view(a * b, c * d)  # Flatten spatial dimensions
        G = torch.mm(features, features.t())  # Compute the dot product of feature maps
        return G.div(a * b * c * d)  # Normalize by the total number of elements

    def forward(self, x):
        """
        Passes input through the model to compute and store the Gram matrices 
        for the target layers, building a structured dictionary.
        """
        self.layer_outputs = {}
        model = nn.Sequential()
        cumulative_grams = []  # List to store cumulative Gram matrices

        for name, layer in self.model.named_children():
            x = layer(x)  # Forward pass through the layer
            if name in self.target_layers:
                gram = self.gram_matrix(x)
                cumulative_grams.append(gram)  # Add current Gram matrix
                self.layer_outputs[name] = cumulative_grams.copy()  # Store copy of cumulative list

        return self.layer_outputs



def reconstruct_image_style(activations_dict):
    """
    Reconstructs an image from the Gram matrices of activations using optimization.

    Parameters:
    ------------
        activations_dict (dict): A dictionary containing the Gram matrices of activations for target layers.

    Returns:
    ------------
        dict
        A dictionary mapping each layer combination to the reconstructed image.
    """

    reconstructed_images = {}

    for layer in style_layers_test:
        print(f"{'-'*10} Running optimization for model up to layer: {layer} {'-'*10}")

        # Initialize a random image to optimize
        reconstructed_image = torch.rand_like(input_image, requires_grad=True)

        # Set up an optimizer for the image
        optimizer = torch.optim.Adam([reconstructed_image], lr=0.01)

        # Optimization loop
        for step in range(3000):
            optimizer.zero_grad()  # Reset gradients
            loss = 0

            # Compute activations for the reconstructed image
            generated_activations = vgg_activations_style(reconstructed_image)

            # Calculate loss for all target layers up to the current one
            for prev_layer in style_layers_test[: style_layers_test.index(layer) + 1]:
                target_gram = activations_dict[prev_layer][style_layers_test.index(prev_layer)]
                generated_gram = generated_activations[prev_layer][style_layers_test.index(prev_layer)]
                loss += torch.nn.functional.mse_loss(generated_gram, target_gram)

            loss /= len(style_layers_test[: style_layers_test.index(layer) + 1])  # Normalize loss

            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the image

            if step % 50 == 0:
                print(f"Step {step}, Loss: {loss.item()}")

        reconstructed_images[layer] = reconstructed_image

    return reconstructed_images

def image_loader(image_name,imshape):
        # scale imported image
        # transform it into a torch tensor
        loader = transforms.Compose([transforms.Resize(imshape),  transforms.ToTensor()])

        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)   # add an additional dimension for fake batch (here 1)
        return image.to(device, torch.float) # move the image tensor to the correct device

def image_display(tensor, title=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()        # clone the tensor
    image = unloader(image.squeeze(0))  # remove the fake batch dimension
    plt.show()
    plt.imshow(image)
    if title is not None:
        plt.title(title)

# Post-process for visualisation purposes
def deprocess(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor * std + mean  # Dénormalisation
    return tensor.clamp(0, 1)

if __name__ == "__main__":

    # ---------- Load the style image ----------
    image_path = "../data/"
    style_image_name = "style.jpeg"

    style_image = image_loader(image_path + style_image_name, imshape=(224,224))
    style_height, style_width = style_image.shape[2], style_image.shape[3]
    print(f"Style image shape : {style_height} x {style_width}")

    # ---------- Load the VGG 19 model with pre-trained weights ----------
    model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device) 

    # ---------- List containing the layers we want to test ----------
    style_layers_test = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"]

    # ---------- Rename the layers of the model ----------
    blocks = [2, 2, 4, 4, 4]  # Number of convolutional layers in each block of the VGG-19 model

    
    renamed_model = nn.Sequential()

    index_conv = 0
    index_relu = 0
    current_block = 0
    i = 0

    for layer in model.features.eval().children():
        # Check if we need to move on to the next block
        if current_block < len(blocks) and index_conv == blocks[current_block]:
            index_conv = 0  # Reinitialize the index for the current block
            current_block += 1

        if isinstance(layer, nn.Conv2d):
            index_conv += 1
            name = f'conv{current_block + 1}_{index_conv}'
            renamed_model.add_module(name, layer)

        elif isinstance(layer, nn.ReLU):
            index_relu += 1
            name = f'relu{current_block + 1}_{index_relu}'
            renamed_model.add_module(name, nn.ReLU(inplace=False))

        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool{current_block + 1}'
            renamed_model.add_module(name, layer)
        i += 1


    # ---------- Preprocess the images for the VGG19 model ----------
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_image = preprocess(torchvision.transforms.functional.to_pil_image(style_image.squeeze(0))).unsqueeze(0)

    # Instantiate the new model 
    vgg_activations_style = VGGActivationsStyle(renamed_model, style_layers_test)
    
    # Get the activations at each layer of the list
    with torch.no_grad():
        activations_style = vgg_activations_style(input_image)
    
    reconstructed_images_style = reconstruct_image_style(activations_dict = activations_style)
    
    for layer, img in reconstructed_images_style.items():
        
        # Deprocess the image
        output_image = deprocess(img.detach())

        # Save the image in a file
        if not os.path.exists("../data/reconstructed_style/"):
            os.makedirs("../data/reconstructed_style/")

        filename = f"../data/reconstructed_style/reconstructed_{layer}.png"
        unloader = transforms.ToPILImage()
        image_to_save = unloader(output_image.squeeze(0))
        image_to_save.save(filename)
        
        # Display the image
        image_display(output_image, layer)
