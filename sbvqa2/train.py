import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters(), lr=0.002)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for v, q, a in tqdm(train_loader, desc='Training'):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, q)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        if (epoch + 1) % 1 == 0:
            model.train(False)
            eval_score, bound = evaluate(model, eval_loader)
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            model.train(True)
            if eval_score > best_eval_score:
                model_path = os.path.join(output, 'model.pth')
                torch.save(model.state_dict(), model_path)
                best_eval_score = eval_score

def evaluate(model, dataloader):
    with torch.no_grad():
        score = 0
        upper_bound = 0
        num_data = 0
        for v, q, a in tqdm(dataloader, desc='Validating'):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            pred = model(v, q)
            batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)

        score = score / len(dataloader.dataset)
        upper_bound = upper_bound / len(dataloader.dataset)
        return score, upper_bound
