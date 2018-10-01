#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 21:21:46 2018
Embedding function
@author: ron
"""
import numpy as np
import pandas as pd

path = "/Users/ron/Desktop/samsung/glove.840B.300d.txt"
words = pd.read_table(path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

dataFile ="atis-2.train.w-intent.iob"
utterances = list()
tags = list()
starts = list()
startid = list()

# reserving index 0 for padding
# reserving index 1 for unknown word and tokens
word_vocab_index = 2
tag_vocab_index = 2
word2id = {'<pad>': 0, '<unk>': 1}
tag2id = {'<pad>': 0, '<unk>': 1}
id2word = ['<pad>', '<unk>']
id2tag = ['<pad>', '<unk>']

utt_count = 0
temp_startid = 0
for line in open(dataFile, 'r'):
	d=line.split('\t')
	utt = d[0].strip()
	t = d[1].strip()
	if len(d) > 2:
		start = np.bool(int(d[2].strip()))
		starts.append(start)
		if start:
			temp_startid = utt_count
		startid.append(temp_startid)
	#print 'utt: %s, tags: %s' % (utt,t)
	temp_utt = list()
	temp_tags = list()
	mywords = utt.split()
	mytags = t.split()
	if len(mywords) != len(mytags):
		print (mywords)
		print (mytags)
	# now add the words and tags to word and tag dictionaries
	# also save the word and tag sequence in training data sets
	for i in range(len(mywords)):
		try:
			ebw = words.loc[mywords[i]]

			if mytags[i] not in tag2id:
				tag2id[mytags[i]] = tag_vocab_index
				id2tag.append(mytags[i])
				tag_vocab_index += 1
			temp_utt.append(ebw)
			temp_tags.append(tag2id[mytags[i]])
		except:
			pass
	utterances.append(temp_utt)
	tags.append(temp_tags)
