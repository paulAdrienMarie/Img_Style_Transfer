import numpy as np
from torchvision import transforms
from pathlib import Path

from style_transfer import style_transfer

def save_image(tensor, save_path=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    
    if save_path:
        image.save(save_path)

# Folder to save the results
output_folder = Path("../data/style_content_ratios")
output_folder.mkdir(parents=True, exist_ok=True)

# Change style or content weights
style_weights = [10 ** 4]
content_weight = 1  

# Content et style images names
content_image_name = "content.jpeg"
style_image_name = "style.jpeg"


for i, style_weight in enumerate(style_weights):
    ratio_name = f"image_ratio_10-{abs(int(np.log10(style_weight)))}"
    print(f"Processing style weight: {style_weight} ({ratio_name})")
    
    output = style_transfer(content_image_name, style_image_name, style_weight, content_weight)
    
    # Saving path
    save_path = output_folder / f"{ratio_name}.jpeg"
    
    save_image(output, save_path=save_path)
