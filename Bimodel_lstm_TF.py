
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:52:42 2018

@author: Xianan Li
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from keras.preprocessing import sequence
import  sys
import pandas as pd
import csv
from keras import layers

path = "/Users/ron/Desktop/samsung/glove.840B.300d.txt"
words = pd.read_table(path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

class dataset():
    def __init__(self,path,time_length):
        self.path = path
        self.time_length = time_length

    def read_data(self):
        utterances = list()
        tags = list()
        intent =list()
        # reserving index 0 for padding
        # reserving index 1 for unknown word and tokens
        tag_vocab_index = 2
        tag2id = {'<pad>': 0, '<unk>': 1}
        id2tag = ['<pad>', '<unk>']
        for line in open(self.path, 'r'):
            #print (line)
            d=line.split('\t')
            utt = d[0].strip()
            t = d[1].strip()
            temp_utt = list()
            temp_tags = list()
            temp_intent = list()
            mywords = utt.split()
            mytags = t.split()
            if len(mywords) != len(mytags):
            	print (mywords)
            	print (mytags)
            # now add the words and tags to word and tag dictionaries
            # also save the word and tag sequence in training data sets
            for i in range(len(mywords)):
                ebw = {}
                try:
                	ebw = words.loc[mywords[i]]
                	if mytags[i] not in tag2id:
                		tag2id[mytags[i]] = tag_vocab_index
                		id2tag.append(mytags[i])
                		tag_vocab_index += 1
                except:
                	pass
                if len(ebw)>0:
                    temp_utt.append(ebw)
                    temp_tags.append(tag2id[mytags[i]])
            temp_intent.append(temp_tags[-1])
     
            if len(temp_utt)>self.time_length:
                temp_utt=temp_utt[0:self.time_length]
                temp_tags=temp_tags[0:self.time_length]
            if len(temp_utt) < self.time_length:
                while len(temp_utt)<self.time_length:
                    temp_utt.append(ebw-ebw)
                    #print (len(temp_utt))
                    temp_tags.append(2)
            #print (len(temp_utt))
            #import pdb
            #pdb.set_trace()
            if len(temp_utt)>0:
                intent.append(temp_intent)
                utterances.append(temp_utt)
                tags.append(temp_tags)
        data = {'utterances': utterances, 'tags': tags,'intent':intent,'id2tag':id2tag,'tag2id':tag2id}
        return data


# =============================================================================
# def get_data():
#
#     training_file ="atis-2.train.w-intent.iob"
#     test_file = 'atis.test.w-intent.iob'
#     # initialization of vocab
#     trainData = dataset(training_file)
#     testData = dataset(test_file)
#     #emptyVocab = {}
#     #emptyIndex = list()
#     #trainData = dataSet(training_file,'train',emptyVocab,emptyVocab,emptyIndex,emptyIndex)
#     #estData = dataSet(test_file, 'test', trainData.getWordVocab(), trainData.getTagVocab(),trainData.getIndex2Word(),trainData.getIndex2Tag())
#
#
#     # preprocessing by padding 0 until maxlen
#     time_length = 20
#     pad_X_train = sequence.pad_sequences(trainData.dataSet['utterances'], maxlen=time_length, dtype='int32', padding='pre')
#     pad_X_test = sequence.pad_sequences(testData.dataSet['utterances'], maxlen=time_length, dtype='int32', padding='pre')
#     pad_y_train = sequence.pad_sequences(trainData.dataSet['tags'], maxlen=time_length, dtype='int32', padding='pre')
#
#
#     # data generation
#     sys.stderr.write("Embedding the input.\n")
#     #X_train =
#     #X_test =
#     #y_train =
#     return X_train,X_test,y_train
# =============================================================================

class bilstm(object):
    def __init__(self,time_length=20,embedding_size=300):
        self.time_length=time_length
        self.embedding_size=embedding_size
            
    def get_output(self,x):
        
        #hidden_size=256
        #dropout_ratio = 0.2
        #batch_size = 64
        #max_epochs =100
        num_hidden = 128
    
        weights = {
                'in' : tf.Variable(tf.random_normal([self.embedding_size,num_hidden])),
                'out': tf.Variable(tf.random_normal([2*num_hidden,self.time_length]))}
        
        biases = {
                'in' : tf.Variable(tf.constant(0.1,shape=[num_hidden,])),
                'out': tf.Variable(tf.constant([self.time_length]))}
        
        x = tf.reshape(x,[-1,self.embedding_size])
        x_in = tf.matmul(x,weights['in'])+biases['in']
        x_in = tf.reshape(x_in,[-1,self.time_length,num_hidden])
        x_in =tf.unstack(x_in,axis=1)
        
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    
        # Get lstm cell output
        outputs, output_state_fw, output_state_bw = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_in,
                                                  dtype=tf.float32)
        import pdb
        pdb.set_trace()
        
        #outputs = tf.stack(outputs,axis=0)
        #outputs=tf.reshape(outputs,[-1,256])
        results = tf.matmul(outputs[-1],weights['out'])+tf.cast(biases['out'],tf.float32)
        return results
        #return tf.matmul(outputs[-1], weights['out']) + biases['out']



def train():
    pass

def test():
    pass

##Data prepration
training_file ="atis-2.train.w-intent.iob"
test_file = 'atis.test.w-intent.iob'
time_lenth =20
embedding_size=300
batch_size = 200
trainData = dataset(training_file,time_lenth)
testData = dataset(test_file,time_lenth)
X_train = trainData.read_data()['utterances']
Y_train = trainData.read_data()['tags']
Y_test = testData.read_data()['tags']
X_test = testData.read_data()['utterances']
Y_train_intent = trainData.read_data()['intent']
Y_test_intent = testData.read_data()['intent']



x=tf.placeholder("float",[batch_size,time_lenth,embedding_size])
y=tf.placeholder("float",[batch_size,time_lenth])

B = bilstm(time_lenth,embedding_size)
result  = B.get_output(x)










