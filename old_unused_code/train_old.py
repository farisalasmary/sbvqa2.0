
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
            Date:  Sep 30, 2021
"""

import torch
import librosa
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from speech_model import *
import pickle
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_mini_batch(questions_by_id, batch_size=64):
    questions_ids = list(questions_by_id.keys())
    num_batches = int(np.ceil(len(questions_ids) / batch_size))
    for i in range(num_batches):
        mini_batch_questions_ids = questions_ids[(i*batch_size) : (i*batch_size + batch_size)]
        
        images_paths, audios_paths, answers_idxs = [], [], []
        for question_id in mini_batch_questions_ids:
            question_dict = questions_by_id[question_id]
            image_path = question_dict['image_path']
            speech_path = question_dict['speech_path']
            answer_idx = question_dict['best_answer_idx']
            
            images_paths.append(image_path)
            audios_paths.append(speech_path)
            answers_idxs.append(int(answer_idx))
        
        yield images_paths, audios_paths, answers_idxs


print('Loading Data...')
train_questions_by_id = pickle.load(open('train_questions_by_id.pkl', 'rb'))
val_questions_by_id = pickle.load(open('val_questions_by_id.pkl', 'rb'))

print('Creating the model...')
images_precalculated_features_path = '/home/farisalasmary/Desktop/speech_vqa/sbvqa_my_implementation/train2014_named_images_features.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

model = SpeechVQA(1000, images_precalculated_features_path=images_precalculated_features_path, device=device)

learning_rate = 0.0003
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 5000
warming_epoch = 3
batch_size = 32
learning_anneal = 1.001

import warnings
warnings.filterwarnings("ignore")


print('Starting Training...')
losses = AverageMeter()
start_epoch = 0
for epoch in range(start_epoch, epochs + 1):
    start_epoch = epoch
    model.train()
    model = model.to(device)
    num_batches = int(np.ceil(len(train_questions_by_id) / batch_size))
    for images_paths, audios_paths, answers_idxs in tqdm(get_mini_batch(train_questions_by_id, batch_size), total=num_batches):
        optimizer.zero_grad()
        
        images_ids = model.read_images(images_paths)
        speech_signals = model.read_audios(audios_paths)
        
        y_pred = model(speech_signals, images_ids)
        y_true = torch.tensor(answers_idxs, requires_grad=False).to(device)
        
        loss = criterion(y_pred, y_true)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        losses.update(loss_value, n=len(images_paths))
        
        del y_pred, y_true
        torch.cuda.empty_cache()
        
        
    print(f"\nTraining loss: {loss_value:0.4f}, epoch: {epoch:4} / {epochs:4}, Avg loss: {losses.avg:.4f}, lr: {learning_rate}")
    # anneal lr
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] / learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))
        learning_rate = g['lr']




