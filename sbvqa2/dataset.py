
import os
import json
import pickle
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train2014', 'val2014'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(None, question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, speech_features_by_question,
                 img_feats_folder='data/img_features',
                 audio_feats_folder='data/audio_features',
                 dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train2014', 'val2014', 'test2015']

        self.dataset_name = name
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        
        self.img_feats_folder = img_feats_folder
        self.audio_feats_folder = audio_feats_folder
        
        self.speech_features_by_question = speech_features_by_question
        self.entries = _load_dataset(dataroot, name)
        self.tensorize()


    def tensorize(self):
        for entry in self.entries:
            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None


    def clean_question(self, text):
        text = text.lower()
        return ' '.join(re.sub(u"[^a-zA-Z ]", "", text,  flags=re.UNICODE).split())

    
    def __getitem__(self, index):
        entry = self.entries[index]
        img_id = entry['image_id']
        img_feats_path = f'{self.img_feats_folder}/COCO_{self.dataset_name}_{img_id:012d}.npz'
        img_feats = np.load(img_feats_path)['x']
        img_feats = torch.from_numpy(img_feats).squeeze(0).float()

        question = self.clean_question(entry['question'])
        quest_feats_filename = self.speech_features_by_question[question]
        quest_feats_path = f'{self.audio_feats_folder}/{quest_feats_filename}'
        question_feats = np.load(quest_feats_path)['x']
        question_feats = torch.from_numpy(question_feats).squeeze(0).float()

        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return img_feats, question_feats, target


    def __len__(self):
        return len(self.entries)


def _collate_fn(batch):
    minibatch_size = len(batch)
    minibatch_visual_features = []
    minibatch_question_features = []
    minibatch_targets = []
    for i in range(minibatch_size):
        sample = batch[i]
        features, question, target = sample
        minibatch_visual_features.append(features)
        minibatch_question_features.append(question)
        minibatch_targets.append(target)

    minibatch_visual_features = pad_sequence(minibatch_visual_features, batch_first=True)
    minibatch_question_features = pad_sequence(minibatch_question_features, batch_first=True)
    minibatch_targets = torch.stack(minibatch_targets, axis=0)
    return minibatch_visual_features, minibatch_question_features, minibatch_targets


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn




