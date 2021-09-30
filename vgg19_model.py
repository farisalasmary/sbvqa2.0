
"""
    @author
          ______         _                  _
         |  ____|       (_)           /\   | |
         | |__ __ _ _ __ _ ___       /  \  | | __ _ ___ _ __ ___   __ _ _ __ _   _
         |  __/ _` | '__| / __|     / /\ \ | |/ _` / __| '_ ` _ \ / _` | '__| | | |
         | | | (_| | |  | \__ \    / ____ \| | (_| \__ \ | | | | | (_| | |  | |_| |
         |_|  \__,_|_|  |_|___/   /_/    \_\_|\__,_|___/_| |_| |_|\__,_|_|   \__, |
                                                                              __/ |
                                                                             |___/
            Email: farisalasmary@gmail.com
            Date:  Sep 22, 2021
"""

import glob
import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models


'''
import torchvision.models as models

model = models.vgg19(pretrained=True)
model.classifier = nn.Sequential(*list(model.classifier.modules())[1:6])
'''


def get_files(mypath, extension='*.txt'):
    return [y for x in os.walk(mypath) for y in glob.glob(os.path.join(x[0], extension))]


def read_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    return image.float()


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
    
    def forward(self, x):
        #return F.normalize(x, p=2, dim=1)
        return F.normalize(x, p=2)


class ImageEncoder(nn.Module):
    def __init__(self, common_embedding_size=512, is_freeze=True):
        super(ImageEncoder, self).__init__()
        self.feature_extractor = models.vgg19(pretrained=True)
        layers_list = list(self.feature_extractor.classifier.modules())[1:6]
        self.feature_extractor.classifier = nn.Sequential(*layers_list)
        last_layer_output_size = list(self.feature_extractor.classifier.named_parameters())[-1][1].shape[0]
        
        self.linear = nn.Linear(last_layer_output_size, common_embedding_size)
        self.l2_norm = L2Norm()
        
        self.is_freeze = is_freeze
        if self.is_freeze:
            self.freeze_feature_extractor()
    
    
    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)
    
    
    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad_(True)
    
    
    def forward(self, x):
        x = self.feature_extractor(x)
        #x = self.linear(x)
        #x = self.l2_norm(x)
        
        return x


    def extract_features(self, x):
        is_training_mode = model.training
        if is_training_mode:
            model.eval()
        
        with torch.no_grad():
            features = self.forward(x)
        
        if is_training_mode:
            model.train()
        
        return features


def get_mini_batchs(imgs_paths, bacth_size=64):
    num_batches = int(np.ceil(len(imgs_paths) / batch_size)) 
    
    for i in range(num_batches):
        mini_batch_imgs_paths = imgs_paths[(i*batch_size):(i*batch_size + batch_size)]
        mini_batch_imgs = [read_image(img_path).cpu().numpy() for img_path in mini_batch_imgs_paths]
        mini_batch_imgs_names = [img_path.strip('.jpg').split('_')[-1] for img_path in mini_batch_imgs_paths]
        mini_batch_imgs = torch.tensor(mini_batch_imgs)
        yield mini_batch_imgs, mini_batch_imgs_names



model = ImageEncoder(512)   
img_path = '/home/farisalasmary/Desktop/speech_vqa/train2014/COCO_train2014_000000487025.jpg'


imgs_folder = '/home/farisalasmary/Desktop/speech_vqa/val2014/'
img = read_image(img_path)

imgs_paths = get_files(imgs_folder, extension='*.jpg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
model = model.to(device)

batch_size = 128
num_batches = int(np.ceil(len(imgs_paths) / batch_size))

named_images_features = {}
for mini_batch_imgs, mini_batch_imgs_names in tqdm(get_mini_batchs(imgs_paths, batch_size), total=num_batches):
    mini_batch_imgs = mini_batch_imgs.to(device)
    extracted_features = model.extract_features(mini_batch_imgs)
    for i, img_features in enumerate(extracted_features):
        named_images_features[mini_batch_imgs_names[i]] = img_features.cpu().clone()
    
    # clean the GPU
    del mini_batch_imgs
    del extracted_features
    torch.cuda.empty_cache()
        

torch.save(named_images_features, 'val2014_named_images_features.pth')








     
        

