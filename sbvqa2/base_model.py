import torch
import torch.nn as nn
from .language_model import QuestionEmbedding
from .fc import FCNet, SimpleClassifier

import pydub
import numpy as np
from torch.autograd import Variable

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.nn.utils.weight_norm import weight_norm
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins
from nemo.core.classes.module import NeuralModule
import nemo.collections.asr as nemo_asr
from torch.nn.utils.rnn import pad_sequence
from .vit import VisionTransformer, interpolate_pos_embed


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )
    return visual_encoder, vision_width

class MyIdentityConvASRDecoder(NeuralModule, Exportable, adapter_mixins.AdapterModuleMixin):
    def forward(self, encoder_output):
        return encoder_output.transpose(1, 2)


class BaseModel(nn.Module):
    def __init__(self, q_dim, v_dim, num_hid, num_classes, rnn_type='GRU', bidirect=False,
                       rnn_layers=1, rnn_dropout=0.0, ans_gen_dropout=0.5):
        super(BaseModel, self).__init__()
        self.q_emb = QuestionEmbedding(q_dim, num_hid, rnn_layers, bidirect, rnn_dropout)
        self.q_net = FCNet([num_hid, num_hid])
        self.v_net = FCNet([v_dim, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, num_classes, ans_gen_dropout)

    def forward(self, v, q):
        q_emb = self.q_emb(q) # batch_size x seqlen x q_dim --> batch_size x q_dim
        v_emb = v # batch_size x v_dim
        q_repr = self.q_net(q_emb) # batch_size x q_dim --> batch_size x num_hid
        v_repr = self.v_net(v_emb) # batch_size x v_dim --> batch_size x num_hid
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr) # batch_size x num_classes
        return logits

class InferenceModel(BaseModel):
    def __init__(self, q_dim, v_dim, num_hid, num_classes, speech_encoder_path, rnn_type='GRU', bidirect=False,
                       rnn_layers=1, rnn_dropout=0.0, ans_gen_dropout=0.5):
        super().__init__(q_dim, v_dim, num_hid, num_classes, rnn_type, bidirect,
                       rnn_layers, rnn_dropout, ans_gen_dropout)
        speech_encoder = nemo_asr.models.EncDecCTCModelBPE.restore_from(speech_encoder_path)
        speech_encoder.decoder = MyIdentityConvASRDecoder()
        self.speech_encoder = speech_encoder
        self.image_size = 480
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.visual_encoder, _ = create_vit(vit='large', image_size=480, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0)

    def convert_to_wav_array(self, input_filepath):
        # VERY IMPORTANT NOTE: for this function to work, you need to install this version of PyDub "pip install git+https://github.com/farisalasmary/pydub.git@master"
        audio_file = pydub.AudioSegment.from_file(input_filepath, format=None, parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000"])
        audio_signal = np.array(audio_file.get_array_of_samples()).astype('float32') / 32767
        return audio_signal

    def load_images(self, imgs_paths, image_size=480):
        images = []
        for img_path in imgs_paths:
            raw_image = Image.open(img_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                ])
            image = transform(raw_image).unsqueeze(0)
            images.append(image)
        images = torch.cat(images, dim=0)
        return image

    def load_audios(self, audio_files):
        audio_signals = []
        audio_signals_lens = []
        for audio_file in audio_files:
            audio_signal = self.convert_to_wav_array(audio_file)
            audio_signal = torch.from_numpy(audio_signal)
            audio_signal_len = len(audio_signal)
            audio_signals.append(audio_signal)
            audio_signals_lens.append(audio_signal_len)
        audio_signals_lens = torch.LongTensor(audio_signals_lens)
        audio_signals = pad_sequence(audio_signals, batch_first=True)
        return audio_signals, audio_signals_lens

    def get_audio_feats(self, audio_signals, audio_signals_lens, batch_size=64):
        device = self.speech_encoder.device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feats, feats_len, _ = self.speech_encoder(
                                input_signal=audio_signals.to(device), input_signal_length=audio_signals_lens.to(device)
                            )
        return feats

    def forward(self, inputs):
        images, audios = inputs
        images = images.to(self.device)
        imgs_feats = self.visual_encoder(images)[:, 0, :] # get the CLS token feats
        audio_signals, audio_signals_lens = audios
        audios_feats = self.get_audio_feats(audio_signals, audio_signals_lens)
        # summarize each question into a single vector
        audios_feats = self.q_emb(audios_feats.to(self.device)) # batch_size x seqlen x q_dim --> batch_size x num_hid
        q_repr = self.q_net(audios_feats) # batch_size x num_hid --> batch_size x num_hid
        v_repr = self.v_net(imgs_feats)  # batch_size x v_dim --> batch_size x num_hid
        joint_repr = q_repr * v_repr # fuse features from both modalities
        logits = self.classifier(joint_repr) # batch_size x num_classes
        return logits
