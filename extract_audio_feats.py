#!apt-get update && apt-get install -y libsndfile1 ffmpeg
#!pip install Cython
#!pip install nemo_toolkit['all']==1.20.0
#!pip uninstall pydub
#!pip install git+https://github.com/farisalasmary/pydub.git@master

import os
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import nemo.collections.asr as nemo_asr
from concurrent.futures import ThreadPoolExecutor
from nemo.core.classes.module import NeuralModule
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins


def get_files(folder, extension='.mp3'):
    paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(root, file))
    return paths


def get_batches(audio_files, batch_size=64):
    num_batches = math.ceil(len(audio_files) / batch_size)
    for i in range(num_batches):
        yield audio_files[i*batch_size : i*batch_size + batch_size]


def get_features(audio_files, asr_model, batch_size=64):
    encoder_output = asr_model.transcribe(audio_files, logprobs=True, batch_size=batch_size, verbose=False)
    return encoder_output


def save_copy(zip_filename, input_folder, output_folder):
    input_folder = input_folder.rstrip('/')
    output_folder = output_folder.rstrip('/')
    os.makedirs(f'{output_folder}/', exist_ok=True)
    os.system(f'zip -qr {output_folder}/{zip_filename}.zip {input_folder}/')


def save_features_npz(x):
    audio_file, audio_features, output_folder = x
    question_id = audio_file.split('/')[-1].replace('.wav', '').replace('.mp3', '')
    output_folder = output_folder.rstrip('/')
    os.makedirs(f'{output_folder}/', exist_ok=True)
    npz_file = f'{output_folder}/{question_id}.npz'
    np.savez_compressed(npz_file, x=audio_features)
    del audio_features


def extract_speech_features(audio_files, output_folder, asr_model, batch_size=64):
    features_per_audio_file = get_features(audio_files, asr_model, batch_size)
    values = []
    for i, audio_file in enumerate(audio_files):
        audio_features = features_per_audio_file[i]
        values.append((audio_file, audio_features, output_folder))
    with ThreadPoolExecutor(max_workers=30) as executor:
        results = executor.map(save_features_npz, values)
    torch.cuda.empty_cache()


class MyIdentityConvASRDecoder(NeuralModule, Exportable, adapter_mixins.AdapterModuleMixin):
    def forward(self, encoder_output):
        return encoder_output.transpose(1, 2)


parser = argparse.ArgumentParser(
                    description='This code uses NeMo implementation of the Conformer encoder to compute audio features',
                    )

parser.add_argument('-i', '--input-audios-folder', required=True, type=str,
                    help='the folder that contains audio files (.wav|.mp3) that will be used to extract features from')
parser.add_argument('-o', '--output-features-folder', required=True, type=str,
                    help='the output folder of .zip file checkpoints that contains audio features saved as .npz per audio file')
parser.add_argument('-m', '--model-path', required=True, type=str,
                    help='NeMo model path with extension .nemo')
parser.add_argument('-t', '--tmp-features-folder', default='speech_features/', type=str,
                    help='temporary folder to save extracted features inside it then it will be compressed and saved in "--output-features-folder" folder')
parser.add_argument('-n', '--batch-size', default=128, type=int,
                    help='batch size to compute features')
parser.add_argument('-R', '--resume-from-batch', default=0, type=int,
                    help='start feature extraction from this batch onward and ignore all batches before')
parser.add_argument('-N', '--save-checkpoint-every-N-batches', default=20, type=int,
                    help='compress the computed features so far and save it in a .zip file in "--output-features-folder". NOTE: this will ALL audio features computed up to this batch, so, always the last checkpoint will contain all computed audio features starting from "--resume-from-batch"')
parser.add_argument('-g', '--gpu', action='store_true',
                    help='whether to use GPU to compute features or not')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
print(f'Device: {device}')

asr_model_path = args.model_path
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(asr_model_path)
original_decoder = asr_model.decoder
asr_model.decoder = MyIdentityConvASRDecoder()

input_audio_folder = args.input_audios_folder
tmp_speech_features_folder = args.tmp_features_folder
zip_files_folder = args.output_features_folder
resume_from_batch = args.resume_from_batch

batch_size = args.batch_size
save_checkpoint_every_n_batches = args.save_checkpoint_every_N_batches

# get ALL .wav and .mp3 files in one list
audio_paths = get_files(input_audio_folder, extension='.wav') + get_files(input_audio_folder, extension='.mp3')
total = math.ceil(len(audio_paths) / batch_size)
for i, batch_audio_files in tqdm(enumerate(get_batches(audio_paths, batch_size)), total=total):
    if i < resume_from_batch:
        continue
    extract_speech_features(batch_audio_files,
                            tmp_speech_features_folder,
                            asr_model,
                            batch_size)
    if (i+1) % save_checkpoint_every_n_batches == 0:
        zip_filename = input_audio_folder.rstrip('/').split('/')[-1] + f'_{i}'
        print(f'Save checkpoint of the features in: {zip_filename}.zip')
        save_copy(zip_filename, tmp_speech_features_folder, zip_files_folder)


zip_filename = input_audio_folder.rstrip('/').split('/')[-1] + f'_{i}_FINAL'
print(f'Save the FINAL checkpoint of the features in: {zip_filename}.zip')
save_copy(zip_filename, tmp_speech_features_folder, zip_files_folder)

