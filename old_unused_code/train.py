
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
import sys


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
            answers_idxs.append(int(answer_idx) - 1) # subtract 1 since the class count start from 1 and we want class 1 to have index 0
        
        yield images_paths, audios_paths, answers_idxs



def evaluate(model, questions_by_id, val_images_precalculated_features_path, batch_size=64, device='cpu'):
    train_precalculated_features = model.image_encoder.precalculated_features
    
    val_precalculated_features = torch.load(val_images_precalculated_features_path)
    model.image_encoder.precalculated_features = val_precalculated_features
    
    total_sum = 0
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        num_batches = int(np.ceil(len(questions_by_id) / batch_size))
        for i, (images_paths, audios_paths, answers_idxs) in enumerate(tqdm(get_mini_batch(questions_by_id, batch_size), total=num_batches)):
            
            images_ids = model.read_images(images_paths)
            speech_signals = model.read_audios(audios_paths)
            
            y_pred = model(speech_signals, images_ids, inference_mode=True)
            y_pred = y_pred.argmax(axis=1)
            
            y_true = torch.tensor(answers_idxs, requires_grad=False).to(device)
            
            total_sum += int((y_pred == y_true).int().sum())
            
            del y_pred, y_true
            torch.cuda.empty_cache()
    
    model.image_encoder.precalculated_features = train_precalculated_features
    
    return total_sum, len(questions_by_id), (total_sum / len(questions_by_id))





print('Loading Data...')
train_questions_by_id = pickle.load(open('train_questions_by_id.pkl', 'rb'))
val_questions_by_id = pickle.load(open('val_questions_by_id.pkl', 'rb'))

print('Creating the model...')

train_images_precalculated_features_path = '/home/farisalasmary/Desktop/speech_vqa/sbvqa_my_implementation/train2014_named_images_features.pth'
val_images_precalculated_features_path = '/home/farisalasmary/Desktop/speech_vqa/sbvqa_my_implementation/val2014_named_images_features.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

model = SpeechVQA(1000, images_precalculated_features_path=train_images_precalculated_features_path, device=device)

learning_rate = 0.0003
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 5000
batch_size = 128
learning_anneal = 1.0001

import warnings
warnings.filterwarnings("ignore")


print('Starting Training...')
losses = AverageMeter()
start_epoch = 1
for epoch in range(start_epoch, epochs + 1):
    start_epoch = epoch
    model.train()
    model = model.to(device)
    num_batches = int(np.ceil(len(train_questions_by_id) / batch_size))
    for i, (images_paths, audios_paths, answers_idxs) in enumerate(tqdm(get_mini_batch(train_questions_by_id, batch_size), total=num_batches)):
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
        
        if i % 50 == 0:
            tqdm.write(f"Training loss: {loss_value:0.4f}, epoch: {epoch:4} / {epochs:4}, Avg loss: {losses.avg:.4f}, lr: {learning_rate}")
    
    if epoch % 5 == 0:
        tqdm.write('Evaluating performance on validation set...')
        total_correct, data_size, accuracy = evaluate(model, val_questions_by_id, val_images_precalculated_features_path, batch_size, device)
        tqdm.write(f'Accuracy on validation set: {accuracy}')
    
    
    # anneal lr
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] / learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))
        learning_rate = g['lr']



