
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:52:42 2018

@author: Xianan Li
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from wordSlotDataSet import dataSet
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers.recurrent import  LSTM
import  sys
from keras.layers.embeddings import Embedding
from Encoding import encoding
from keras.layers import Input, merge, concatenate, Dense, Dropout, Activation, RepeatVector, Permute, Reshape, RepeatVector, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD

class BiLstm(object):

	def __init__(self, input_vocab_size,output_vocab_size):
		self.hidden_size=256
		self.embedding_size = 100
		self.input_vocab_size = input_vocab_size
		self.output_vocab_size = output_vocab_size
		self.time_length=20
		self.dropout_ratio = 0.2
		self.batch_size = 64
		self.max_epochs =100

	def build(self):
		raw_current = Input(shape=(self.time_length,), dtype='int32')
		print (np.shape(raw_current))
		current = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=self.time_length, mask_zero=False)(raw_current)
		print (np.shape(current))
		fencoder = LSTM(self.hidden_size, return_sequences=False)(current)
		bencoder = LSTM(self.hidden_size, return_sequences=False, go_backwards=True)(current)
		flabeling = LSTM(self.hidden_size, return_sequences=True)(current)
		blabeling = LSTM(self.hidden_size, return_sequences=True, go_backwards=True)(current)
		encoder = concatenate([fencoder, bencoder],axis =-1)
		labeling = concatenate([flabeling, blabeling],axis =-1)
		print (np.shape(labeling))
		encoder = RepeatVector(self.time_length)(encoder)
		tagger = concatenate([encoder, labeling])
		#3tagger = Dropout(self.dropout_ratio)(tagger)
		print('size is')
		print (np.shape(tagger))
		tagger = Flatten()(tagger)
		print (np.shape(tagger))
		prediction = TimeDistributed(Dense(self.output_vocab_size, activation='softmax'))(tagger)
		self.model = Model(input=raw_current, output=prediction)
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
		self.model.compile(loss='categorical_crossentropy',optimizer=sgd)

	def train(self, xtrain,ytrain):
		self.model.fit(xtrain, ytrain, batch_size=self.batch_size, nb_epoch=self.max_epochs, verbose=1)





training_file ="atis-2.train.w-intent.iob"
test_file = 'atis.test.w-intent.iob'
# initialization
emptyVocab = {}
emptyIndex = list()
trainData = dataSet(training_file,'train',emptyVocab,emptyVocab,emptyIndex,emptyIndex)
testData = dataSet(test_file, 'test', trainData.getWordVocab(), trainData.getTagVocab(),trainData.getIndex2Word(),trainData.getIndex2Tag())


time_length = 20
pad_X_train = sequence.pad_sequences(trainData.dataSet['utterances'], maxlen=time_length, dtype='int32', padding='pre')
pad_X_test = sequence.pad_sequences(testData.dataSet['utterances'], maxlen=time_length, dtype='int32', padding='pre')
pad_y_train = sequence.pad_sequences(trainData.dataSet['tags'], maxlen=time_length, dtype='int32', padding='pre')
num_sample_train, max_len = np.shape(pad_X_train)
num_sample_test, max_len = np.shape(pad_X_test)

# encoding input vectors
input_vocab_size = trainData.getWordVocabSize()
output_vocab_size = trainData.getTagVocabSize()

# data generation
input_type = 'embedding'    #options: 1hot, embedding, predefined
sys.stderr.write("Vectorizing the input.\n")
X_train = encoding(pad_X_train, input_type, time_length, input_vocab_size)
X_test = encoding(pad_X_test, input_type, time_length, input_vocab_size)
y_train = encoding(pad_y_train, 'embedding', time_length, output_vocab_size)

bimodel  = BiLstm(input_vocab_size,output_vocab_size)
print('build')
bimodel.build()
print('train')
bimodel.train(X_train,y_train)
