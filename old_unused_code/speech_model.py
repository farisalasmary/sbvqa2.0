
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
            Date:  Sep 23, 2021
"""

import torch
import librosa
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import torchaudio
#import torchaudio.transforms

class L2Norm(nn.Module):
    def __init__(self, dim):
        super(L2Norm, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class SpeechEncoder(nn.Module):
    def __init__(self, common_embedding_size=512, speech_precalculated_features_path=None, device='cpu'):
        super(SpeechEncoder, self).__init__()
        
        speech_precalculated_features_path = speech_precalculated_features_path # TODO: implement precalculated speech features
        
        conv1_out_channels = 32
        conv2_out_channels = 64
        conv3_out_channels = 128
        conv4_out_channels = 256
        conv5_out_channels = 512 # TODO: you may set this equal to "common_embedding_size" since it is the output
                                 # of the last conv layer that is fed to lstm model

        conv1_kernel_size = 64
        conv2_kernel_size = 32
        conv3_kernel_size = 16
        conv4_kernel_size = 8
        conv5_kernel_size = 4
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, conv1_out_channels, conv1_kernel_size, stride=2),
            nn.BatchNorm1d(conv1_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(conv1_out_channels, conv2_out_channels, conv2_kernel_size, stride=2),
            nn.BatchNorm1d(conv2_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(conv2_out_channels, conv3_out_channels, conv3_kernel_size, stride=2),
            nn.BatchNorm1d(conv3_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(conv3_out_channels, conv4_out_channels, conv4_kernel_size, stride=2),
            nn.BatchNorm1d(conv4_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(conv4_out_channels, conv5_out_channels, conv5_kernel_size, stride=2),
            # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/5
            nn.BatchNorm1d(conv5_out_channels), # this layer may result in an error if the input size is 1 example
            nn.ReLU()
        )
        
        hidden_layer_size = common_embedding_size
        self.lstm = nn.LSTM(common_embedding_size, hidden_layer_size, batch_first=True, num_layers=1)
        
        self.l2_norm = L2Norm(2)
        
        self.linear = nn.Linear(hidden_layer_size, common_embedding_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x): # x is a speech signal
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) # output dims: batch_size x num_features x sequence_length
                          # num_features: is the number of filters in the last conv layer
        
        x = x.transpose(1, 2) # batch_size x sequence_length x num_features
        
        x = self.lstm(x)[0][:, -1, :]  # take the last output vector. This is similar to return_sequences=False in keras LSTM
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

################################################################################################################
################################################################################################################
################################################################################################################

class ImageEncoder(nn.Module):
    def __init__(self, common_embedding_size=512, is_freeze=True, precalculated_features_path=None, device='cpu'):
        super(ImageEncoder, self).__init__()
        self.feature_extractor = models.vgg19(pretrained=True)
        layers_list = list(self.feature_extractor.classifier.modules())[1:6]
        self.feature_extractor.classifier = nn.Sequential(*layers_list)
        last_layer_output_size = list(self.feature_extractor.classifier.named_parameters())[-1][1].shape[0]
        
        self.device = device
        
        self.l2_norm = L2Norm(1)
        self.linear = nn.Linear(last_layer_output_size, common_embedding_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        
        self.precalculated_features = None
        if not precalculated_features_path is None:
            self.precalculated_features = torch.load(precalculated_features_path)
        
        self.is_freeze = is_freeze
        if self.is_freeze:
            self.freeze_feature_extractor()
    
    
    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)
    
    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad_(True)
    
    def get_precalculated_features(self, images_ids):
        # this function extracts features from a predefined dict of images features
        images_features = []
        for image_id in images_ids:
            images_features.append(self.precalculated_features[image_id].cpu().numpy())
        
        
        images_features = torch.tensor(images_features, requires_grad=False).to(self.device)
        
        return images_features
    
    def forward(self, x):
        # x is expected to be a 4D PyTorch tensor of dimensions: batch_size x # of channels x height x width
        # which represents a list of RGB images 
        if not self.precalculated_features:
            x = self.feature_extractor(x)
        else:
            # if you want to use precalculated features of some images, i.e., for training
            # you can pass x as a normal list of images IDs. See "self.get_precalculated_features"
            # for more information
            x = self.get_precalculated_features(x)
        
        x = self.l2_norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        
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


################################################################################################################
################################################################################################################
################################################################################################################


class SpeechVQA(nn.Module):
    def __init__(self, output_size, common_embedding_size=512, images_precalculated_features_path=None, speech_precalculated_features_path=None, device='cpu'):
        super(SpeechVQA, self).__init__()
        
        self.device = device
        
        self.images_precalculated_features_path = images_precalculated_features_path
        self.speech_precalculated_features_path = speech_precalculated_features_path
        self.device = device
        self.image_encoder = ImageEncoder(common_embedding_size, precalculated_features_path=images_precalculated_features_path, device=device).to(self.device)
        self.speech_encoder = SpeechEncoder(common_embedding_size, speech_precalculated_features_path=speech_precalculated_features_path, device=device).to(self.device)
        self.l2_norm = L2Norm(1)
        self.linear1 = nn.Linear(common_embedding_size, common_embedding_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(common_embedding_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def _join_features(self, speech_features, image_features):
        return speech_features * image_features
    
    
    def forward(self, speech_signals, images, inference_mode=False):
        image_features = self.image_encoder(images).to(self.device)
        speech_features = self.speech_encoder(speech_signals).to(self.device)
        
        joint_features = self._join_features(speech_features, image_features)
        normlaized_features = self.l2_norm(joint_features)
        dense = self.linear1(normlaized_features)
        dense = self.activation(dense)
        dense = self.dropout(dense)
        
        output = self.linear2(dense)
        
        # get the probability distribution over all classes during the inference time becuase we will use
        # the nn.CrossEntropyLoss criterion which combines LogSoftmax and NLLLoss in one single class, hence,
        # no need to apply nn.Softmax during training
        if inference_mode:
            output = self.softmax(output)
        
        return output
    
    
    def read_images(self, images_paths):
        if type(images_paths) != list:
            raise ValueError('You MUST provide a list of paths of images....')
        
        # if the we use a precalculated features, just return a list of images ids
        if not self.images_precalculated_features_path is None:
            images_ids = []
            for image_path in images_paths:
                # extract the image id from 
                image_id = image_path.split('/')[-1].lower().split('_')[-1].replace('.jpg', '')
                images_ids.append(image_id)
            
            return images_ids # return the ids instead
        
        images = []
        for image_path in images_paths:
            image = Image.open(image_path).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            image = transform(image).cpu().numpy()
            images.append(image)
        return torch.tensor(images).float().to(self.device)
    
    def load_audio_faster(self, audio_path, resample_rate=16000):
        signal, sample_rate = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(sample_rate, resample_rate)
        resampled_signal = resampler(signal)
        
        return resampled_signal
    
    def read_audios(self, audios_paths):
        if type(audios_paths) != list:
            raise ValueError('You MUST provide a list of paths of audio files....')
        
        max_signal_len = -1
        audios_signals = []
        for audio_path in audios_paths:
            #audio_signal, sr = librosa.load(audio_path, sr=16_000)
            #audio_signal = audio_signal.reshape(1, -1)
            audio_signal = self.load_audio_faster(audio_path, resample_rate=16000).cpu().numpy()
            
            audios_signals.append(audio_signal)
            max_signal_len = max(max_signal_len, audio_signal.shape[1])
        
        tmp = []
        for audio_signal in audios_signals:
            len_diff = max_signal_len - audio_signal.shape[1]
            if len_diff > 0:
                audio_signal = np.concatenate([audio_signal, np.zeros(len_diff).reshape(1, -1)], axis=1)
            
            tmp.append(audio_signal)
        
        audios_signals = tmp
        
        return torch.tensor(audios_signals).float().to(self.device)



'''

from speech_model import *
images_precalculated_features_path = '/home/farisalasmary/Desktop/speech_vqa/sbvqa_my_implementation/train2014_named_images_features.pth'

audio_path = '/home/farisalasmary/volume/arabic_kaldi_models/NEOM_30_sec.wav'
image_path = '/home/farisalasmary/Desktop/speech_vqa/train2014/COCO_train2014_000000487025.jpg'

model = SpeechVQA(1000, images_precalculated_features_path=images_precalculated_features_path)

images_paths = [image_path]*10 # just a single file for simplcity
audios_paths = [audio_path]*10 # just a single file for simplcity
images_ids = model.read_images(images_paths)
speech_signals = model.read_audios(audios_paths)
out = model(speech_signals, images_ids)


'''



'''
if __name__ == '__main__':

from speech_model import *
images_precalculated_features_path = '/home/farisalasmary/Desktop/speech_vqa/sbvqa_my_implementation/train2014_named_images_features.pth'
model = SpeechVQA(1000, images_precalculated_features_path=images_precalculated_features_path)
audio_path = '/home/farisalasmary/volume/arabic_kaldi_models/NEOM_30_sec.wav'
image_path = '/home/farisalasmary/Desktop/speech_vqa/train2014/COCO_train2014_000000487025.jpg'

audios_paths = [audio_path]*20 # just a single file for simplcity
#images_paths = [image_path]*20 # just a single file for simplcity

#images = model.read_images(images_paths)
#speech_signals = model.read_audios(audios_paths)

audios_paths = [audio_path]*10 # just a single file for simplcity
speech_signals = model.read_audios(audios_paths)
images_ids = ['000000022098', '000000312896', '000000544731', '000000024257', '000000187230', '000000359141', '000000035668', '000000497746', '000000215972', '000000434139']

#out = model(speech_signals, images)
out = model(speech_signals, images_ids)



'''




