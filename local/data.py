# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 02:10:25 2018

@author: Bohyun
"""

from konlpy.tag import Twitter

import re

pos_tagger = Twitter()

emoji_pattern = re.compile("["
u"\U0001F600-\U0001F64F"  # emoticons
u"\U0001F300-\U0001F5FF"  # symbols & pictographs
u"\U0001F680-\U0001F6FF"  # transport & map symbols
u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                   "]+", flags=re.UNICODE)

def tokenize(doc):

    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]



def read_raw_data(filename):

    with open(filename, 'r', encoding='utf-8') as f:

        print('loading data')

        data = []
        newscore = []

        for line in f.read().splitlines():
            data_temp = emoji_pattern.sub(r'', line)
            
            
         #   print(data_temp)
            data.append(data_temp.split('\t'))
    #    data = [line.split('\t') for line in f.read().splitlines()]

     
        
        print('pos tagging to token')

        #data = [(tokenize(row[1]), int(row[2])) for row in data[0:]]
        data = [(tokenize(row[1]), int(row[2]),int(row[3]),str(row[4]),str(row[5])) for row in data[0:]]
        #data score 3 class로 바꿈
              #  lidata = list(data)
        for i in range(len(data)):
            if data[i][1] in [1]:
                newscore.append(0)
                #lidata[i][1] = 0
            elif data[i][1] in [2, 3]:
                newscore.append(1)
                #lidata[i][1] = 1
            elif data[i][1] in [4, 5]:
                newscore.append(2)
               
                #lidata[i][1] = 2
        
       # data = tuple(lidata)
     #   print(len(newscore))
        #data[1]
        

    return data, newscore
  #  return data



def build_vocab(tokens):

    print('building vocabulary')

    vocab = dict()

    vocab['#UNKOWN'] = 0

    vocab['#PAD'] = 1

    for t in tokens:

        if t not in vocab:

            vocab[t] = len(vocab)

    return vocab



def get_token_id(token, vocab):

    if token in vocab:

        return vocab[token]

    else:

        0 # unkown



def build_input(data, vocab, newscore):



    def get_onehot(index, size):

        onehot = [0] * size

        onehot[index] = 1

        return onehot



        
    print('building input')

    result = []
    i = 0
    for d in data:

        sequence = [get_token_id(t, vocab) for t in d[0]]

        while len(sequence) > 0:
           # i = i+1
            seq_seg = sequence[:60]

            sequence = sequence[60:]



            padding = [1] *(60 - len(seq_seg))

            seq_seg = seq_seg + padding

            
            result.append((seq_seg, get_onehot(newscore[i], 3)))
#            result.append((seq_seg, get_onehot(d[1], 2)))
         #   print(i)
        i = i+1


    return result 



def save_data(filename, data):

    def make_csv_str(d):

        output = '%d' % d[0]

        for index in d[1:]:

            output = '%s,%d' % (output, index)

        return output



    with open(filename, 'w', encoding='utf-8') as f:
        i = 0
        for d in data:

            data_str = make_csv_str(d[0])

            label_str = make_csv_str(d[1])

            f.write (data_str + '\n')

            f.write (label_str + '\n')
            i = i+1


def save_vocab(filename, vocab):

    with open(filename, 'w', encoding='utf-8') as f:

        for v in vocab:

            f.write('%s\t%d\n' % (v, vocab[v]))

            

def load_data(filename):

    result = []

    with open(filename, 'r', encoding='utf-8') as f:

        lines = f.readlines()

        for i in range(int(len(lines)/2)):

            data = lines[i*2]

            label = lines[i*2 + 1]



            result.append(([int(s) for s in data.split(',')], [int(s) for s in label.split(',')]))

    return result



def load_vocab(filename):

    result = dict()

    with open(filename, 'r', encoding='utf-8') as f:

        for line in f.readlines():

            ls = line.split('\t')

            result[ls[0]] = int(ls[1])



    return result





if __name__ == '__main__':

 
    data, newscore = read_raw_data('mm.txt')

    
    tokens = [t for d in data for t in d[0]]

    vocab = build_vocab(tokens)

    d = build_input(data, vocab, newscore)

    save_data('test_data.txt', d)

    save_vocab('test_vocab.txt', vocab)



    d2 = load_data('test_data.txt')

    vocab2 = load_vocab('test_vocab.txt')



    assert(len(d2) == len(d))

    for i in range(len(d)):

        assert(len(d2[i]) ==  len(d[i]))

        for j in range(len(d[i])):

            assert(d2[i][j] == d[i][j])



    for index in vocab:

        assert(vocab2[index] == vocab[index])

    