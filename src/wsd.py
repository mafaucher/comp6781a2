#!/usr/bin/env python

from __future__ import division
import os.path
import itertools, operator
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus.reader import senseval

def import_senseval_corpus(corpus):
	"""
	corpus should be 'train' or 'test'
	words is a list of words in the corpus
	"""
	assert corpus in ['train', 'test']
	train = senseval.SensevalCorpusReader(corpus, ["EnglishLS."+corpus])
	word_instances = {}
	for instance in train.instances():
		if not word_instances.has_key(instance.word):
			word_instances[instance.word] = []
		word_instances[instance.word].append(instance)
	return word_instances

def most_common(instances):
	senses = [instance.senses[0] for instance in instances]
	groups = itertools.groupby(sorted(senses))
	def _auxfun((item, iterable)):
		return len(list(iterable)), -senses.index(item)
	return max(groups, key=_auxfun)[0]

train_corpus = import_senseval_corpus('train')
test_corpus = import_senseval_corpus('test')

def wsd_features(instance):
	"""
	Defines feature vector for a word
	"""
	features = {'predecessor' : instance.context[instance.position-1]}
	return features

def test(word):
	"""
	Train Naive Bayes Classifier and 
	"""
	instances = train_corpus[word]
	featuresets = [(wsd_features(instance), instance.senses[0]) for instance in instances]
	classifier = NaiveBayesClassifier.train(featuresets)
	
	total = 0
	accuracy = 0
	baseline = 0
	bl = most_common(train_corpus[word])
	for instance in train_corpus[word]:
		c = classifier.classify(wsd_features(instance))
		if c == bl:
			baseline += 1
		if c == instance.senses[0]:
			accuracy += 1
		total+=1
	print "word:", word
	print "accuracy:", accuracy/total
	print "baseline:", baseline/total
	print "="*80

for word in train_corpus:
	test(word)
