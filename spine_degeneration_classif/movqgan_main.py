from movqgan import get_movqgan_model
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys


def prepare_image(img):
    """Transform and normalize PIL Image to tensor."""
    transform = T.Compose(
        [
            T.RandomResizedCrop(
                512,
                scale=(1.0, 1.0),
                ratio=(1.0, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
        ]
    )
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))


def show_images(batch, return_image=False):
    """Display a batch of images inline."""
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    if return_image:
        return Image.fromarray(reshaped.numpy())
    plt.imshow(Image.fromarray(reshaped.numpy()))
    plt.show()


img_path = "270M/examples.png"
img = prepare_image(Image.open(img_path))
# show_images(img.unsqueeze(0))


weights = torch.load(
    "270M/movqgan_270M.ckpt", map_location=torch.device("cpu"), weights_only=True
)
model = get_movqgan_model("270M", pretrained=False, device="cpu")
with open("model_structure.txt", "w") as f:
    sys.stdout = f
    print(model)
exit()
with torch.no_grad():
    random_output = model(img.unsqueeze(0))[0]
    print("Random output", random_output.shape)
    model.load_state_dict(weights)
    real_output = model(img.unsqueeze(0))[0]
    print("Real output")

show_images(random_output)
show_images(real_output)
