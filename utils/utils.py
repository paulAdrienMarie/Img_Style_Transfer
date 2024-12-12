from tkinter import Image
from matplotlib import transforms
import matplotlib.pyplot as plt
from torch import device
import torch


def image_loader(image_name, image_shape=(224,224)):
    # scale imported image
    # transform it into a torch tensor
    loader = transforms.Compose([transforms.Resize(image_shape),  transforms.ToTensor()])

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