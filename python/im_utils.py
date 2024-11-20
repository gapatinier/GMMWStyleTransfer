from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as transforms
import torch

def tensor_imshow(tensor, title=None, ax=None):
    # Convert tensor to numpy and transpose to HWC format
    inp = tensor.detach().cpu().numpy().transpose((1, 2, 0))  # Use .cpu() to move tensor to CPU if it's on a GPU
    inp = np.clip(inp, 0, 1)  # Clip values to [0, 1]
    
    # If no axis is provided, create one
    if ax is None:
        ax = plt.gca()
    
    # Show the image
    ax.imshow(inp)
    
    # Remove axis labels and ticks
    ax.axis('off')
    
    # If title is provided, set it
    if title is not None:
        ax.set_title(title)
    
    # Ensure interactive mode is off
    if plt.isinteractive():
        plt.ioff()
# import torch
# tensor_imshow(torch.randn(3, 256, 512), 'pytorch tensor')


def load_img(path, new_size):
    img = Image.open(path).convert(mode='RGB')
    if new_size:
        # for fixed-size squared resizing, leave only the following line uncommented in this if statement
        img = transforms.resize(img, (new_size, new_size), Image.BICUBIC)
        width, height = img.size
        max_dim_ix = np.argmax(img.size)
        if max_dim_ix == 0:
            new_shape = (int(new_size * (height / width)), new_size)
            img = transforms.resize(img, new_shape, Image.BICUBIC)
        else:
            new_shape = (new_size, int(new_size * (width / height)))
            img = transforms.resize(img, new_shape, Image.BICUBIC)
    return transforms.to_tensor(img)

def alpha_show(tfc, fc, decoder, style=None):
    # List of alphas for interpolation
    alpha = [0, 0.2, 0.5, 0.8, 1]
    alphas = torch.tensor(alpha).view(-1, 1, 1, 1)
    fdiff = (tfc-fc).unsqueeze(0).repeat(5, 1, 1, 1)
    atfcontent = fc.unsqueeze(0).repeat(5, 1, 1, 1) + alphas*fdiff
    tcontent = decoder(atfcontent.float())
    p = 1
    if style is None:
        p = 0
    # Create a subplot grid with 1 row and len(alphas) columns
    fig, axes = plt.subplots(1, len(alpha)+p, figsize=(15, 5))

    # Loop through each alpha value
    for i in range(5):
        
        # Plot the result on the corresponding subplot
        ax = axes[i]
        tensor_imshow(tcontent[i], ax=ax)  # Pass the axis to the tensor_imshow function
        ax.set_title(f'Alpha = {alpha[i]}')  # Set title for each plot


    if not(style is None):
        ax = axes[5]
        tensor_imshow(style, ax=ax)  # Pass the axis to the tensor_imshow function
        ax.set_title(f'Style')  # Set title for each plot

    # Show the plots
    plt.tight_layout()
    plt.show()