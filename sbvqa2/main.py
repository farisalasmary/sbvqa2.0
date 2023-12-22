import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import VQAFeatureDataset, AudioDataLoader
from base_model import BaseModel
from train import train
import utils
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-img-feats-folder', type=str, default='data/blip_feats_train2014/')
    parser.add_argument('--val-img-feats-folder', type=str, default='data/blip_feats_val2014/')
    parser.add_argument('--audio-feats-folder', type=str, default='data/speech_features/')
    parser.add_argument('--dataroot', type=str, default='data/')
    parser.add_argument('--feature-by-question-file', type=str, default='data/features_by_question.pkl')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--num-hid', type=int, default=1024)
    parser.add_argument('--num-classes', type=int, default=3129)
    parser.add_argument('--q-dim', type=int, default=512, help='The dimension of the input question vector/matrix')
    parser.add_argument('--v-dim', type=int, default=1024, help='The dimension of the input visual vector/matrix')
    parser.add_argument('--output', type=str, default='models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    train_img_feats_folder = args.train_img_feats_folder
    val_img_feats_folder = args.val_img_feats_folder
    audio_feats_folder = args.audio_feats_folder
    print('Load "speech_features_by_question"....')
    speech_features_by_question = pickle.load(open(args.feature_by_question_file, 'rb'))
    print('Loading was successfully completed!!!')

    print('Load "train_dset"....')
    train_dset = VQAFeatureDataset('train2014', speech_features_by_question,
                                   train_img_feats_folder, audio_feats_folder, args.dataroot)
    print('Loading was successfully completed!!!')

    print('Load "eval_dset"....')
    eval_dset = VQAFeatureDataset('val2014', speech_features_by_question,
                                  val_img_feats_folder, audio_feats_folder, args.dataroot)
    print('Loading was successfully completed!!!')

    batch_size = args.batch_size
    model = BaseModel(args.q_dim, args.v_dim, args.num_hid, args.num_classes,
                      rnn_type='GRU', bidirect=False, rnn_layers=1,
                      rnn_dropout=0.0, ans_gen_dropout=0.5).cuda()
    print('Preparing train_loader...')
    train_loader = AudioDataLoader(train_dset, batch_size, shuffle=True, num_workers=8)
    print('Preparing eval_loader...')
    eval_loader =  AudioDataLoader(eval_dset, batch_size, shuffle=False, num_workers=8)
    train(model, train_loader, eval_loader, args.epochs, args.output)
