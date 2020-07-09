import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prepro import Vocabulary
from dataset import s2vtDataset, collate_fn
from model import s2vtModel, EncoderRNN, DecoderRNN
from customloss import CustomLoss
from trainer import Trainer
import logging

def main():
    training_avi_feats = './training_data/feat'
    training_label = './training_label.json'
    testing_avi_feats = './testing_data/feat'
    testing_label = './testing_label.json'
    print('start building dictionary')
    dictionary = Vocabulary(PATH=training_label, min_cnt=5)
    print('end building dictionary')
    train_dataset = s2vtDataset(training_avi_feats, training_label, dictionary)
    print(train_dataset.__len__())
    test_dataset = s2vtDataset(testing_avi_feats, testing_label, dictionary)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    INPUT_FEAT_DIM = 4096
    HIDDEN_SIZE = 256
    WORD_DIM = 256
    OUTPUT_DIM = len(dictionary.vocab) + 4

    EPOCH = 200
    MDL_OUTDIR = 'model'
    if not os.path.exists(MDL_OUTDIR):
        os.mkdir(MDL_OUTDIR)
    encoder = EncoderRNN(input_size=INPUT_FEAT_DIM, hidden_size=HIDDEN_SIZE)
    decoder = DecoderRNN(hidden_size=HIDDEN_SIZE,
                         output_size=OUTPUT_DIM,
                         vocab_size=OUTPUT_DIM,
                         word_dim=WORD_DIM
                         )
    model = s2vtModel(encoder=encoder, decoder=decoder)
    trainer = Trainer(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, dictionary=dictionary)

    s = time.time()

    for epoch in range(EPOCH):
        print('epoch: {}'.format(epoch))
        trainer.train()
        trainer.eval()

    e = time.time()

    torch.save(model, "{}/{}.h5".format(MDL_OUTDIR, 'test'))

    print("Finished training {}  Time elapsed: {: .3f} seconds. \n".format('test', e-s))
if __name__ == '__main__':
    main()
