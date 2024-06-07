from pydicom import dcmread
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from tqdm import tqdm
import json


def get_dcm_array(path):
    arr = dcmread(path).pixel_array
    # print(arr)
    # print(arr.shape)
    # print(arr.max(), arr.min())

    # plt.imshow(arr, cmap="gray")
    # plt.show()
    return arr


def get_torchvision_vectors(arr):
    """
    Transforms a DICOM image array into a format suitable for a pre-trained
    Vision Transformer (ViT) model and returns the encoded feature vector.

    Args:
        arr (numpy.ndarray): Pixel array of the DICOM image.

    Returns:
        torch.Tensor: Encoded feature vector from the ViT model.
    """

    # image transformation
    shape = (224, 224)
    arr_temp = torch.from_numpy(arr).to(torch.float)
    arr_temp = torch.stack([arr_temp, arr_temp, arr_temp])
    arr_temp = arr_temp.view(1, *arr_temp.shape)
    arr_temp = torchvision.transforms.Resize(shape)(arr_temp)

    # Load pre-trained ViT model and remove classification head
    model = torchvision.models.vit_b_16(weights="DEFAULT")
    model.heads = nn.Identity()
    
    # get feature vector from model
    enc = model(arr_temp)
    return enc


if __name__ == "__main__":
    root = "../data/rsna-2024-lumbar-spine-degenerative-classification/"
    path = os.path.join(root, "train_images/*/*/*.dcm")
    images = glob(path)
    
    steps = 1000
    final_index = np.ceil(len(images) / steps).astype(int)
    print(f"Total images: {len(images)}, Final index: {final_index}")

    # Process images in batches and save feature vectors to JSON files
    for i in range(131, -1, -1):
    # for i in range(final_index + 1, -1, -1):
        output = []
        for img in tqdm(images[i * steps : (i + 1) * steps]):
            arr = get_dcm_array(img)
            enc = get_torchvision_vectors(arr)
            output.append(
                {
                    "image_path": img,
                    "encoding_vit_b_16": enc.tolist(),
                }
            )
        # Save the output to a JSON file
        with open(f"../data/train_data_{i}.json", "w") as f:
            json.dump(output, f, indent=4)
