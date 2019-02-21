# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 02:11:12 2018

@author: Bohyun
"""

from data import *

from textcnn import TextCNN

import tensorflow as tf

import numpy as np



TRAIN_FILENAME = 'mm.txt'

TRAIN_DATA_FILENAME = TRAIN_FILENAME + '.data'

TRAIN_VOCAB_FILENAME = TRAIN_FILENAME + '.vocab'



SEQUENCE_LENGTH = 60

NUM_CLASS = 3



def test():

    with tf.Session() as sess:

        vocab = load_vocab(TRAIN_VOCAB_FILENAME)

        cnn = TextCNN(SEQUENCE_LENGTH, NUM_CLASS, len(vocab), 128, [3,4,5], 128)

        saver = tf.train.Saver()

        saver.restore(sess, './textcnn.ckpt')

        print('model restored')

        
        while 1:
            
            input_text = input('사용자 평가를 문장으로 입력하세요(Z 입력시 종료): ')
            if input_text in ['z', 'Z']:
                break
            tokens = tokenize(input_text)
    
            print('입력 문장을 다음의 토큰으로 분해:')
    
            print(tokens)
    
    
    
            sequence = [get_token_id(t, vocab) for t in tokens]
    
            x = []
    
            while len(sequence) > 0:
    
                seq_seg = sequence[:SEQUENCE_LENGTH]
    
                sequence = sequence[SEQUENCE_LENGTH:]
    
    
    
                padding = [1] *(SEQUENCE_LENGTH - len(seq_seg))
    
                seq_seg = seq_seg + padding
    
    
    
                x.append(seq_seg)
    
            
    
            feed_dict = {
    
                cnn.input : x,
    
                cnn.dropout_keep_prob : 1.0
    
            }
    
    
            #별점 예측
            predict = sess.run([cnn.predictions], feed_dict)
            result = np.array(predict)
            result = result[0][0]
            print("=========================결과==========================")
            print("별점: ", result)
            
            if result in [0]:
                print("불만족")
            elif result in [1]:
                print("보통")
            elif result in [2]:
                print("만족")
                
                
                
                

if __name__ == '__main__':

    test()