import os 
import re
import json
class Vocabulary():
    '''
    Vocabulary is like a dictionary of this task. 
    '''
    def __init__(self, PATH, min_cnt=3):
        self.file_path = PATH 
        self.min_cnt = min_cnt 
        self.word_cnt = {}
        self.vocab = None
        self.rare_words = None 
        self.w2i = {} 
        self.i2w = {}
        self.initialize()
        self.build_dict()
    def initialize(self):
        labels = json.load(open(self.file_path, 'r'))
        for label in labels:
            captions = label['caption']
            for caption in captions:
                caption = caption.replace('.','').split()
                for word in caption:
                    self.word_cnt[word] = self.word_cnt.get(word, 0) + 1
        self.vocab = [w for w in self.word_cnt if self.word_cnt.get(w, 0) >= self.min_cnt]
        self.rare_words = [w for w in self.word_cnt if self.word_cnt.get(w, 0) < self.min_cnt]
    def process_caption(self, sentence):
        stopwords = [',',';','.','?','!']
        for word in stopwords:
            sentence = sentence.replace(word, '')
        sentence = sentence.split()    
        sentence = ['<bos>'] + [w  if self.word_cnt.get(w, 0)  > self.min_cnt \
                else '<unk>' for w in sentence ] + ['<eos>']
        return sentence
    def build_dict(self):
        self.w2i['<pad>'] = 0
        self.w2i['<bos>'] = 1
        self.w2i['<eos>'] = 2
        self.w2i['<unk>'] = 3
        self.i2w[0] = '<pad>'
        self.i2w[1] = '<bos>'
        self.i2w[2] = '<eos>'
        self.i2w[3] = '<unk>'
        for index, word in enumerate(self.vocab):
            self.w2i[word] = index + 4 
            self.i2w[index+4] = word 
    def sent2ind(self, sentence):
        return [self.w2i[word] for word in sentence]
    def ind2sent(self, index):
        #import IPython
        #IPython.embed()
        return [self.i2w[int(i)] for i in index]
if __name__ == '__main__':
    a = Vocabulary('./training_label.json', 2)
    print(len(a.vocab))
    print(len(a.rare_words))
    print(a.word_cnt['pen'])
    print(a.process_caption("A chicken is being seasoned."))

    aa = a.process_caption("A chicken is being seasoned.")

    bb = a.sent2ind(aa)
    print(bb)

    cc = a.ind2sent(bb)
    print(cc)
