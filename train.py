
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
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

for epoch in range(start_epoch, epochs + 1):
    start_epoch = epoch
    model.train()
    model = model.to(device)
    for i, text in enumerate(train_data['text'].sample(frac=1)):
        optimizer.zero_grad()        
        y_pred_tensor = model([text])
        y_tensor = get_y_tensor([train_data['sentiment'].iloc[i]], device)
        loss = criterion(y_pred_tensor, y_tensor)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        losses.update(loss_value)
        
        #del y_pred_tensor, y_tensor
        #torch.cuda.empty_cache()
        
        if i % 5000 == 0:
            print(i)
        i += 1
    print(f"\nTraining loss: {loss_value:0.4f}, iteration: {i+1:4} / {len(train_data):4}, epoch: {epoch:4} / {epochs:4}, Avg loss: {losses.avg:.4f}, lr: {learning_rate}")
    # anneal lr
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] / learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))
        learning_rate = g['lr']
    
