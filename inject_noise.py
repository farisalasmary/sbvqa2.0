
# this script samples a random noise file, then it uses the same sampled noise file
# to augment the same audio question using different noise SNRs in dB
# I did it this way since I wanted to see the effect of using different SNRs levels
# for the same audio file and the same noise file in different speakers to be fair in the comparisons later

# !apt-get update && apt-get install -y libsndfile1 ffmpeg
# !pip install audiomentations==0.28.0

# Run example:
# python inject_noise.py -i in_dir/ -I UrbanSound8K/ -o out_dir/ -l 0,5,10,15,20 -j 6

import os
import math
import json
import random
import logging
import argparse
import warnings
from tqdm import tqdm
from scipy.io import wavfile
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map  # or thread_map
from audiomentations import AddBackgroundNoise, PolarityInversion
from audiomentations.core.audio_loading_utils import load_sound_file

random.seed(29)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)


def get_files(folder, extension='.mp3'):
    paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(root, file))
    return paths


def inject_noise(sound_file_path, noise_file_path, output_file_path, snr_in_db):
    samples, sample_rate = load_sound_file(sound_file_path, sample_rate=16_000, mono=True)
    transform = AddBackgroundNoise(
        sounds_path=noise_file_path,
        min_snr_in_db=snr_in_db,
        max_snr_in_db=snr_in_db,
        noise_transform=PolarityInversion(),
        p=1.0
    )
    augmented_samples = transform(samples=samples, sample_rate=sample_rate)
    wavfile.write(output_file_path, rate=sample_rate, data=augmented_samples)


def _inject_func_multiproc(func_input):
    sound_file_path, noise_file_path, output_file_path, snr_db_value = func_input
    inject_noise(sound_file_path, noise_file_path, output_file_path, snr_db_value)
    return (sound_file_path, noise_file_path, output_file_path)


parser = argparse.ArgumentParser(
                    description='This code uses injects random noise files into each audio file in a given folder',
                    )

parser.add_argument('-i', '--input-audios-folder', required=True, type=str,
                    help='the folder that contains clean audio files (.wav|.mp3)')
parser.add_argument('-I', '--input-noises-folder', required=True, type=str,
                    help='the folder that contains noise files (.wav|.mp3) where a random noise file will be selected from for injection')
parser.add_argument('-o', '--output-folder', required=True, type=str,
                    help='the output folder of the injected files')
parser.add_argument('-l', '--noise-levels', default='0,5,10,15,20', type=str, 
                    help='a list of noise levels separated by comma with no spaces in between, e.g., "0,5,10,15,20"')
parser.add_argument('-j', '--num-jobs', default=8, type=int,
                    help='number of concurrent jobs')
parser.add_argument('-n', '--batch-size', default=128, type=int,
                    help='number of files per process')

args = parser.parse_args()

sound_audios_folder = args.input_audios_folder
noise_audios_folder = args.input_noises_folder
out_folder = args.output_folder

noise_files = get_files(noise_audios_folder, extension='.wav')
mp3_files = get_files(sound_audios_folder, extension='.mp3')
wav_files = get_files(sound_audios_folder, extension='.wav')
sound_file_paths = mp3_files + wav_files

mp3_files = sorted(mp3_files, key=lambda x: x.split('/')[-1].replace('.mp3', '').replace('.wav', ''))
snr_db_values = [int(i) for i in args.noise_levels.strip().split(',')] #[0, 5, 10, 15, 20]

prepared_args = []
for sound_file_path in tqdm(sound_file_paths):
    noise_file_path = random.choice(noise_files)
    for snr_db_value in snr_db_values:     
        output_folder = f'{out_folder}/snr_db_{snr_db_value}'
        os.makedirs(output_folder, exist_ok=True)
        audio_filename = sound_file_path.strip().split('/')[-1].replace('.mp3', '').replace('.wav', '')
        output_file_path = f'{output_folder}/{audio_filename}.wav'
        prepared_args.append((sound_file_path, noise_file_path, output_file_path, snr_db_value))


r = process_map(_inject_func_multiproc, prepared_args, max_workers=args.num_jobs)
json.dump(r, open(f'{out_folder}/input_n_noise_n_output_files.json', 'w'))
#with Pool(args.num_jobs) as p:
#    r = list(tqdm(p.imap(_inject_func_multiproc, sound_file_paths, chunksize=args.batch_size), total=len(sound_file_paths)))