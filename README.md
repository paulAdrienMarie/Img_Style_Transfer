# Image Style Transfer

### Authors: Paul-Adrien Marie & Hugo Boulenger

---

### Project Overview

This project implements neural style transfer based on the method presented by **Leon A. Gatys**, **Alexander S. Ecker** and **Matthias Bethge** in their paper, *"Image Style Transfer Using Convolutional Neural Networks" (CVPR, 2016)*. The goal is to create a stylized image by blending the content of one image with the artistic style of another. This approach optimizes an image to balance content and style elements.

---

### Key Components

1. **VGG-19 Model**: We use the pretrained VGG-19 CNN (available via torchvision) to extract content and style features.

2. **Loss Functions**:
   - **Content Loss**: Quantifies the difference between the feature representations of the content image and the stylized image at specific layers.
   - **Style Loss**: Based on Gram matrices, this loss captures the texture of the style image by comparing the correlations between features in different layers.
   - **Total Loss**: Combines the weighted content and style losses to guide the optimization, with adjustable content and style weights to fine-tune the results.

3. **Optimization Process**:
   - We frame style transfer as an optimization task. Starting with a random or content-initialized image, the algorithm iteratively updates this image to minimize the total loss using the L-BFGS optimizer.