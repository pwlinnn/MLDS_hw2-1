import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import logging
import IPython

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hid_dim*1) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
    def forward(self, hidden_state, encoder_outputs):
        """
        @param hidden: hidden states of decoder, shape (1, batch size, dim) 
        """
        batch_size, src_len, feat_dim = encoder_outputs.shape 
        hidden_state = hidden_state.view(batch_size, 1, feat_dim).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden_state, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2) # attention: [batch size, src_len]
        return F.softmax(attention, dim=1)
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        """ 
        Todo: add Dropout Layer
        """
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, input):
        """
        @param input: input sequence of shape (b, 80, 4096), where b = batch size
        @returns
        """
        bs, seq_len, feature_dim = input.shape
        input = self.embedding(input)
        input = self.dropout(input)
        output, h_n = self.gru(input)
        return output, h_n 

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dictionary=None, dropout_p=0.1, threshold=0.5):
        """
        @param hidden_size: equivalent to hid_dim, dimension of hidden state, in this task we let enc_hid_dim == dec_hid_dim
        @param output_size 
        @param vocab_size: the amount of valid word (word_cnt > min_cnt) in the dictionary
        @param output_size: == vocab_size
        @param threshold: scheduled sampling ratio 
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.output_size = output_size 
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.dictionary = dictionary
        self.threshold = threshold

        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.gru = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size, hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, processed_captions=None, mode='train'):
        """
        @param encoder_last_hidden_state: last hidden state of encoder, shape (1, batch_size, hidden_size)
        @param encoder_output: hidden state of all time step, shape (batch_size, src_len, hidden_size)
        @param processed_captions: groundtruth of the target sentences
        """
        batch_size = encoder_last_hidden_state.shape[1]
        decoder_current_hidden_state = encoder_last_hidden_state
        decoder_current_input_word = torch.ones(batch_size,1).long() # decoder input word should be <bos> in the beginning, w2i['<bos>'] = 1, #<bos> (batch x word index)
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []
        
        processed_captions = self.embedding(processed_captions) # processed_captions: shape (batch, max_tgt_len, word_dim)
        tgt_len = processed_captions.shape[1]
        for i in range(tgt_len-1): 
            ### scheduled sampling
            groundtruth = processed_captions[:,i,:] # cur input words of all time steps, shape (batch_size, 1, word_dim)
            cur_input_words = groundtruth if random.uniform(0.05, 0.995) > self.threshold else self.embedding(decoder_current_input_word)
            ### end scheduled sampling
            # cur_input_words shape (batch_size, 1, word_dim)
            cur_input_words = cur_input_words.squeeze(1)
            logger.debug('encoder output size: {}'.format(encoder_output.shape))
            logger.debug('decoder current hidden state: {}'.format(decoder_current_hidden_state.shape))

            a = self.attention(decoder_current_hidden_state, encoder_output) # a shape (batch size, src len)
            a = a.unsqueeze(1) # a shape (batch size, 1, src len)
            weighted = torch.bmm(a, encoder_output).squeeze(1) # weighted shape (batch size, 1, hidden_size)
            
            logger.debug('current input word size: {}'.format(cur_input_words.shape))
            logger.debug('weighted size: {}'.format(weighted.shape))
            #IPython.embed()
            dec_input = torch.cat((weighted, cur_input_words), dim=1).unsqueeze(1) # dec_input shape (batch_size, 1, hidden_size+word_dim)
            
            logger.debug('decoder input size: {}'.format(dec_input.shape))
            logger.debug('decoder current hidden state size: {}'.format(decoder_current_hidden_state.shape))

            gru_output, decoder_current_hidden_state = self.gru(dec_input, decoder_current_hidden_state)

            logger.debug('gru output size: {}'.format(gru_output.shape))
            logger.debug('decoder output hidden state size: {}'.format(decoder_current_hidden_state.shape))

            logProb = self.to_final_output(gru_output.squeeze(1)) # logprob shape (batch, output_size)
            seq_logProb.append(logProb.unsqueeze(1))
            decoder_current_input_word = torch.max(logProb, dim=1)[1].view(-1,1)
        
        seq_logProb = torch.cat(seq_logProb, dim=1) # seq_logProb shape (batch_size, tgt_len-1, output_size)

        logger.debug('total output seq size: {}'.format(seq_logProb.size()))

        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
    def infer(self, encoder_last_hidden_state, encoder_output):
        batch_size = encoder_last_hidden_state.shape[1]
        decoder_current_hidden_state = encoder_last_hidden_state
        decoder_current_input_word = torch.ones(batch_size,1).long() # decoder input word should be <bos> in the beginning, w2i['<bos>'] = 1, #<bos> (batch x word index)
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []
        assume_tgt_len = 28
        for i in range(assume_tgt_len-1):
            cur_input_words = self.embedding(decoder_current_input_word)
            cur_input_words = cur_input_words.squeeze(1)
            a = self.attention(decoder_current_hidden_state, encoder_output) # a shape (batch size, src len)
            a = a.unsqueeze(1) # a shape (batch size, 1, src len)
            weighted = torch.bmm(a, encoder_output).squeeze(1) # weighted shape (batch size, 1, hidden_size)
            dec_input = torch.cat((weighted, cur_input_words), dim=1).unsqueeze(1) # dec_input shape (batch_size, 1, hidden_size+word_dim)
            gru_output, decoder_current_hidden_state = self.gru(dec_input, decoder_current_hidden_state)

            logProb = self.to_final_output(gru_output.squeeze(1)) # logprob shape (batch, output_size)
            seq_logProb.append(logProb.unsqueeze(1))
            decoder_current_input_word = torch.max(logProb, dim=1)[1].view(-1,1)

        seq_logProb = torch.cat(seq_logProb, dim=1) # seq_logProb shape (batch_size, tgt_len-1, output_size)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

class s2vtModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(s2vtModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, avi_feats, mode, target_sentences=None):
        """
        @param avi_feat: shape (batch_size, 80, 4096)
        @param mode: 'train' or 'infer'
        @param target_sentences: groundtruth
        """
        encoder_output, encoder_last_hidden_state = self.encoder(avi_feats)

        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state, encoder_output, processed_captions=target_sentences, mode='train')
        elif mode == 'infer':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state, encoder_output)
        return seq_logProb, seq_predictions
if __name__ == '__main__':
    import logging
    #logger.setLevel(logging.WARNING)
    from prepro import Vocabulary

    json_file = './testing_label.json'
    numpy_file = './testing_data/feat'

    dictionary = Vocabulary(json_file, min_cnt=5)



    input_data = torch.randn(3, 80, 4096).view(-1, 80, 4096)

    encoder = EncoderRNN(input_size=4096, hidden_size=1000)
    decoder = DecoderRNN(hidden_size=1000, output_size=1700, vocab_size=1700, word_dim=128, dictionary=dictionary)

    model = s2vtModel(encoder=encoder, decoder=decoder)

    ground_truth = torch.rand(3, 27).long()

    for step in range(50, 100):
        seq_prob, seq_predict = model(avi_feats=input_data, mode='train', target_sentences=ground_truth)

        if step % 10 == 0:
            print(seq_prob.size())
            print(seq_predict.size())        