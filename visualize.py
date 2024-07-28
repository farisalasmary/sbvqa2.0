
from my_gradcam import MyGradCAM
import numpy as np
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
import requests
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def reshape_transform(tensor, height=30, width=30):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visualize_gradcam(model, image_path, audio_path):
    target_layers = [model.visual_encoder.blocks[-1].norm1]
    cam = MyGradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    images = model.load_images([image_path])
    signals_with_lens = model.load_audios([audio_path])
    input_tensor = (images, signals_with_lens)
    model.eval()
    model.q_emb.rnn.train() # This is important to solve the following error:
                            # RuntimeError: cudnn RNN backward can only be called in training mode
                            # Since dropout = 0 in rnn, we can guarantee that the output will not
                            # change in each forward pass
    targets = None
    eigen_smooth = False
    aug_smooth = False
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=eigen_smooth,
                        aug_smooth=aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    model.q_emb.rnn.eval()
    image_size = (480, 480)
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, image_size)
    rgb_img = np.float32(rgb_img) / 255
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    viz_img_path = image_path.replace('.jpg', '_my_gradcam.jpg')
    cv2.imwrite(viz_img_path, cam_image)
    return viz_img_path


def get_attention(model, image_path):
    patch_size = 16
    img = model.load_images([image_path])
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    register_blk = len(model.visual_encoder.blocks) - 1
    image_embeds = model.visual_encoder(img.to(model.device), register_blk=register_blk)
    attentions = model.visual_encoder.blocks[-1].attn.get_attention_map()
    nh = attentions.shape[1]  # number of head
    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].detach().cpu().numpy()
    return attentions

def prepare_n_save_attention(image_paths, model):
    image_data = []
    for j, image_path in enumerate(image_paths):
        attention = get_attention(model, image_path)
        # filename = image_path.split('/')[-1].split('.')[0]
        filename = image_path.replace('.jpg', '')
        img = Image.open(image_path).resize((480, 480))
        white_img = (np.ones((480,480,3)) * 255).astype(int)
        n_heads = attention.shape[0]
        for i in range(n_heads+2):
            if i == 0:
                image_data.append(img)
            elif i != 0 and i < 9:
                # axarr[i, 0].imshow(attention[i-1], cmap='inferno')
                image_data.append(attention[i-1])
            elif i == 9:
                image_data.append(white_img)
            elif i != 0 and i > 9:
                # axarr[i, 0].imshow(img, cmap='inferno')
                image_data.append(attention[i-2])
    num_images = len(image_paths)
    fig = plt.figure(figsize=(num_images * 2 * 8., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(num_images * 2, 9),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    a = []
    for i in range(1, 18):
        if i == 9:
            a.append('')
        elif i < 9:
            a.append(f'Head {i}')
        elif i > 9:
            a.append(f'Head {i-1}')
    titles = ['Image'] + a # [f'Head {i}' if i != 9 else '' for i in range(17)]

    i = 0
    for title, ax, im in zip(titles, grid, image_data):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        if i <= 9:
            ax.set_title(title)
        else:
            ax.set_title(title, y=-0.2)
        ax.axis('off')
        i += 1
    # plt.show()
    viz_img_path = f'{filename}_all_heads.png'
    plt.savefig(viz_img_path, dpi=200, bbox_inches='tight')
    return viz_img_path


def visualize_attention(model, image_path):
    return prepare_n_save_attention([image_path], model)

