import os
import sys
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from customloss import CustomLoss
from tqdm import tqdm
class Trainer():
    """
    Workflow for training 1 epoch
    """
    def __init__(self, model, train_dataloader=None, test_dataloader=None, dictionary=None):
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.__CUDA__ = torch.cuda.is_available()

        if self.__CUDA__:
            self.model = model.cuda()
        else:
            self.model = model.cpu()

        self.loss_fn = CustomLoss()
        self.cumloss = 0
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        self.dictionary = dictionary
    def train(self):
        self.model.train()
        cnt = 0
        cumloss = 0
        print('start training')
        #for i, batch in tqdm(enumerate(self.train_loader)):
        for i, batch in enumerate(self.train_loader):
            print('batch no. {}'.format(i+1))
            cnt += 1
            avi_feats, processed_captions, lengths = batch
            if self.__CUDA__:
                avi_feats, processed_captions = avi_feats.cuda(), processed_captions.cuda()
            #avi_feats.requires_grad = True
            #processed_captions.requires_grad = True

            self.optimizer.zero_grad()
            seq_logProb, seq_predictions = self.model(avi_feats, 'train', processed_captions)

            # There won't be <bos> in the prediction, therefore I remove <bos> from all processed_captions(groundtruth)
            processed_captions = processed_captions[:,1:]
            loss = self.loss_fn(seq_logProb, processed_captions, lengths)
            loss.backward()
            self.optimizer.step()
            cumloss += float(loss)
        print('training avgloss: {}'.format(cumloss/cnt))
    
    def eval(self):
        self.model.eval()
        test_predictions, test_truth = None, None

        for i, batch in enumerate(self.test_loader):
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()
            seq_logProb, seq_predictions = self.model(avi_feats, mode='infer')
            ground_truths = ground_truths[:, 1:]

            test_predictions = seq_predictions[:3]
            test_truth = ground_truths[:3]
            break

        result = [' '.join(self.dictionary.ind2sent(s)) for s in test_predictions]
        print('Testing Result: \n{} \n{}\n{}\n'.format(result[0], result[1], result[2]))
        truth = [' '.join(self.dictionary.ind2sent(s)) for s in test_truth]
        print('Ground Truth: \n{} \n{}\n{}\n'.format(truth[0], truth[1], truth[2]))
    
    def test(self):
        self.model.eval()
        ss = []

        for batch_idx, batch in enumerate(self.test_loader):
            # prepare data
            id, avi_feats = batch

            if self.__CUDA__:
                avi_feats = avi_feats.cuda()

            id, avi_feats = id, Variable(avi_feats).float()

            # start inferencing process
            seq_logProb, seq_predictions = self.model(avi_feats, mode='infer')
            test_predictions = seq_predictions


            result = [[x if x != '<UNK>' else 'something' for x in self.dictionary.ind2sent(s)] for s in test_predictions]
            result = [' '.join(s).split('<EOS>')[0] for s in result]

            rr = zip(id, result)

            for r in rr:
                ss.append(r)
        return ss
