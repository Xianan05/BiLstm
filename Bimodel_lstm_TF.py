
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


def get_data():
    
     training_file ="atis-2.train.w-intent.iob"
     test_file = 'atis.test.w-intent.iob'
     # initialization of vocab
     trainData = dataset(training_file)
     testData = dataset(test_file)
     #emptyVocab = {}
     #emptyIndex = list()
     #trainData = dataSet(training_file,'train',emptyVocab,emptyVocab,emptyIndex,emptyIndex)
     #estData = dataSet(test_file, 'test', trainData.getWordVocab(), trainData.getTagVocab(),trainData.getIndex2Word(),trainData.getIndex2Tag())
    
    
     # preprocessing by padding 0 until maxlen
     time_length = 20
     pad_X_train = sequence.pad_sequences(trainData.dataSet['utterances'], maxlen=time_length, dtype='int32', padding='pre')
     pad_X_test = sequence.pad_sequences(testData.dataSet['utterances'], maxlen=time_length, dtype='int32', padding='pre')
     pad_y_train = sequence.pad_sequences(trainData.dataSet['tags'], maxlen=time_length, dtype='int32', padding='pre')
    
    
     # data generation
     sys.stderr.write("Embedding the input.\n")
     #X_train =
     #X_test =
     #y_train =
     return X_train,X_test,y_train

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

        #outputs = tf.stack(outputs,axis=0)
        #outputs=tf.reshape(outputs,[-1,256])
        #results = tf.matmul(outputs[-1],weights['out'])+tf.cast(biases['out'],tf.float32)
        
        results=tf.stack(outputs,1)
        return results
    
class decoder(object):
    def __init__(self,time_length = 20,embedding_size=300):
        self.time_length=time_length
        self.embedding_size=embedding_size
    def get_slot(self,x1,x2):
        num_hidden=128
        x = tf.concat([x1,x2],2)
        weights = {
                'out': tf.Variable(tf.random_normal([2*num_hidden,self.time_length]))}
        
        biases = {
                'out': tf.Variable(tf.constant([self.time_length]))}
      
        x=tf.unstack(x,axis = 1)
        lstm_cell = rnn.BasicLSTMCell(2*num_hidden, forget_bias=1.0)
        outputs,states = rnn.static_rnn(lstm_cell,x, dtype = tf.float32)
        results = tf.matmul(outputs[-1], weights['out'])+tf.cast(biases['out'],tf.float32)

        return results
    
    def get_intent(self,x1,x2):
        num_hidden=128
        x = tf.concat([x1,x2],2)
        weights = {
                'out': tf.Variable(tf.random_normal([2*num_hidden,1]))}
        
        biases = {
                'out': tf.Variable(tf.constant([self.embedding_size]))}
      
        x=tf.unstack(x,axis = 1)
        lstm_cell = rnn.BasicLSTMCell(2*num_hidden, forget_bias=1.0)
        outputs,states = rnn.static_rnn(lstm_cell,x, dtype = tf.float32)
        results = tf.matmul(outputs[-1], weights['out'])+tf.cast(biases['out'],tf.float32)
        results = tf.Print(results,[results],'intent value is ')
        return results
        
class batch(object):
    def __init__(self,num, x, y,intent):
        self.batch_size = num
        self.x = x
        self.y = y
        self.intent = intent
        self.idx = np.arange(0 , np.shape(x)[0])
        #np.random.shuffle(self.idx)
    
    def next_batch(self):
        if len(self.idx>self.batch_size):
            np.random.shuffle(self.idx)
            idx1 = self.idx[:self.batch_size]
            x_shuffle = list()
            y_shuffle = list()
            intent_shuffle=list()
            for i in idx1:
                x_shuffle.append(self.x[i])
                y_shuffle.append(self.y[i])
                intent_shuffle.append(self.intent[i])
            #self.idx = self.idx[self.batch_size :]
        return x_shuffle, y_shuffle, intent_shuffle

def test():
    pass

##Data prepration
training_file ="atis-2.train.w-intent.iob"
test_file = 'atis.test.w-intent.iob'
time_lenth =20
embedding_size=300
batch_size = 200
max_iter = 100
lr= 0.001
 
trainData = dataset(training_file,time_lenth)
testData = dataset(test_file,time_lenth)
X_train = trainData.read_data()['utterances']
Y_train = trainData.read_data()['tags']
Y_test = testData.read_data()['tags']
X_test = testData.read_data()['utterances']
Y_train_intent = trainData.read_data()['intent']
Y_test_intent = testData.read_data()['intent']


tf.reset_default_graph()
x=tf.placeholder("float",[batch_size,time_lenth,embedding_size])
y=tf.placeholder("float",[batch_size,time_lenth])
intent = tf.placeholder("float",[batch_size,1])

with tf.variable_scope('f1'):
    f_1 = bilstm(time_lenth,embedding_size)
    h1  = f_1.get_output(x)
with tf.variable_scope('f2'):
    f_2 = bilstm(time_lenth,embedding_size)
    h2 = f_2.get_output(x)
with tf.variable_scope('g1'):  
    g_1 = decoder(time_lenth,embedding_size)
    s1 = g_1.get_intent(h1,h2)
    s1 = tf.Print(s1,[s1],'intent is ')
  
with tf.variable_scope('g2'):  
    g_2 = decoder(time_lenth,embedding_size)
    s2 = g_2.get_slot(h1,h2)

intent_cost = tf.losses.mean_squared_error(s1,intent)
slot_cost = tf.losses.mean_squared_error(s2,y)
train_intent = tf.train.AdamOptimizer(lr).minimize(intent_cost)
train_slot = tf.train.AdamOptimizer(lr).minimize(slot_cost)
train_op = tf.group(train_intent, train_slot)


accuracy_intent = tf.losses.mean_squared_error(s1 ,intent)
accuracy_slot = tf.losses.mean_squared_error(s2,y)
acc = tf.group(accuracy_intent,accuracy_slot)

with tf.Session() as sess:
    if (int((tf.__version__).split('.')[1])<12 and int((tf.__version__).split('.')[0]))<1:
        init = tf.initialize_all_initializer()
    else:
        init = tf.global_variables_initializer()
    
    sess.run(init)
    train_batch = batch(batch_size, X_train, Y_train,Y_train_intent)
    step = 0
    while step < max_iter*batch_size:
        b_x,b_y,b_i = train_batch.next_batch()
        sess.run([train_op],feed_dict ={x:b_x, y:b_y, intent:b_i})
        if step % 20 == 0:
            print('===step--'+str(step)+'===')
            print(sess.run(acc,feed_dict={x:b_x, y:b_y, intent:b_i}))
            
        step += 1
            
        
        






