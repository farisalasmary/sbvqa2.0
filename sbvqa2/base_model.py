import torch
import torch.nn as nn
from language_model import QuestionEmbedding
from fc import FCNet, SimpleClassifier


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
