import os
import sys
import json
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from prepro import Vocabulary
import random

class s2vtDataset(Dataset):
    def __init__(self, video_feat_npy_path, label_json, dictionary):
        self.label_json = label_json
        #self.video_feat = np.load(video_feat_npy_path) # shape: (80, 4096)
        self.data_pair = []
        self.avi_dict = {}
        self.dictionary = dictionary
        print('start opening json file')
        labels = json.load(open(self.label_json, 'r'))
        print('opened training labels')
        print(len(labels))
        for i, label in enumerate(labels):
            print('label no. {}'.format(i))
            captions = label['caption']
            _id = label['id']
            for caption in captions:
                caption = self.dictionary.process_caption(caption)
                caption = self.dictionary.sent2ind(caption)
                self.data_pair.append((_id, caption))
         
        file_names = os.listdir(video_feat_npy_path)
        print(len(file_names))
        for i, file_name in enumerate(file_names):
            print('file no. {}'.format(i)) 
            path = os.path.join(video_feat_npy_path, file_name)
            key = file_name[:-4]
            self.avi_dict[key] = np.load(path)
    def __getitem__(self, idx):
        (_id, caption) = self.data_pair[idx] # id: avi file name
        data = torch.Tensor(self.avi_dict[_id]) 
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(caption)

    def __len__(self):
        return len(self.data_pair)
def collate_fn(data):
    '''
    @param data: (avi, captions)
    @returns avi: the same avi file returned from __getitem__()
    @returns processed_captions: the captions with paddings.
    @returns lengths: valid length of each caption 
    '''
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi, captions = list(zip(*data))
    lengths = [len(cap) for cap in captions] 
    avi = torch.stack(avi, dim=0)
    processed_captions = torch.zeros(len(captions), max(lengths)).long()
    for i, caption in enumerate(captions):
        processed_captions[i,:lengths[i]] = caption[:lengths[i]]
    return avi, processed_captions, lengths
if __name__ == '__main__':
    torch.manual_seed(0)
    import time
    from torch.autograd import Variable

    json_file = './testing_label.json'
    numpy_file = './testing_data/feat'

    helper = Vocabulary(json_file, min_cnt=5)

    dataset = s2vtDataset(label_json=json_file,
            video_feat_npy_path=numpy_file, dictionary=helper)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=collate_fn)

    ss = time.time()

    for epoch in range(1):

        s = time.time()

        print('epoch: {}'.format(epoch+1))
        for batch_n, batch in enumerate(dataloader):
            #e = time.time()

            #print('batch No.{} time loading batch: {}'.format(batch_n, e-s))

            #s = time.time()
            print('batch no: {}'.format(batch_n))
            data, label, lengths = batch
            print(label[:, :12])
            print(lengths)


            for s in label:
                print(helper.ind2sent(s))


            #packed = pack_padded_sequence(input=label, lengths=lengths, batch_first=True)
            #
            #print(packed.data)
            #
            #checkpoint()


            break
        e = time.time()

        #print('time for one epoch: {}'.format(e-s))

    ee = time.time()
