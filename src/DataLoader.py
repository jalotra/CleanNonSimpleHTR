from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocessor as preprocess


class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath

class FilePaths:
    """ Filenames and paths to data """
    fnCharList = '../model/charList.txt'
    fnWordCharList = '../model/wordCharList.txt'
    fnCorpus = '../model/corpus.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/testImage1.png'  ## path to recognize the single image

class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


class DataLoader:
	"loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

	def __init__(self, filePath, batchSize, imgSize, maxTextLen, load_aug = True):
		"loader for dataset at given location, preprocess images and text according to parameters"

		assert filePath[-1]=='/'

		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = []
	
		f=open(filePath+'words.txt')
		chars = set()
		bad_samples = []
		bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
		for line in f:
			# ignore comment line
			if not line or line[0]=='#':
				continue
			
			lineSplit = line.strip().split(' ')
			assert len(lineSplit) >= 9
			
			# filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
			fileNameSplit = lineSplit[0].split('-')
			fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'
			
			# GT text are columns starting at 9
			gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
			chars = chars.union(set(list(gtText)))
			print(os.path.exists(fileName), fileName,  gtText)
			# check if image is not empty
			if not os.path.getsize(fileName):
				bad_samples.append(lineSplit[0] + '.png')
				continue

			# put sample into list
			self.samples.append(Sample(gtText, fileName))

		# FOR THE CVL DATABASE
		f = open("../data/" + "cvl_words.txt")
		for line in f:
			# print(line)
			fileName = "../data/" + "cvl_words/" + line
			# print(fileName)
			# print(fileName)
			gtText = self.truncateLabel(str(line.split(".")[0].split("-")[-1]), maxTextLen)
			# remove unwanted chars
			gtText = self.change_cvl_database_chars(gtText)
			
			print(os.path.exists(fileName), fileName,  gtText)
			if(len(gtText) == 0):
				continue
			# print(gtText)
			chars = chars.union(set(list(gtText)))
			self.samples.append(Sample(gtText, fileName))

		# some images in the IAM dataset are known to be damaged, don't show warning for them
		if set(bad_samples) != set(bad_samples_reference):
			print("Warning, damaged images found:", bad_samples)
			print("Damaged images expected:", bad_samples_reference)
		
		# Shuffle the samples 
		random.shuffle(self.samples)

		# split into training and validation set: 90% - 10%
		splitIdx = int(0.90 * len(self.samples))
		test_set_examples = int(0.10*len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]
		# self.testSamples = self.samples[splitIdx+test_set_examples : ]
		# put words into lists
		self.trainWords = [x.gtText for x in self.trainSamples]
		self.validationWords = [x.gtText for x in self.validationSamples]
		# self.testWords = [x.gtText for x in self.testSamples]

		# number of randomly chosen samples per epoch for training 
		self.numTrainSamplesPerEpoch = 25000 
		
		# start with train set
		self.trainSet()

		# list of all chars in dataset
		self.charList = sorted(list(chars))

	def change_cvl_database_chars(self, str):
		m = {
			"ä" : "a",
			"ü" : "u",
			"ö" : "o"
		}
		final = ""
		for i , ch in enumerate(str):
			if ch in m:
				final += m[ch]
			else:
				final += ch

		return final


	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text


	def trainSet(self):
		"switch to randomly chosen subset of training set"
		self.dataAugmentation = True
		self.currIdx = 0
		random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]


	def validationSet(self):
		"switch to validation set"
		self.dataAugmentation = False
		self.currIdx = 0
		self.samples = self.validationSamples

	# def testSet(self):
	# 	self.dataAugmentation = False
	# 	self.currIdx = 0
	# 	self.samples = self.testSamples


	def getIteratorInfo(self):
		"current batch index and overall number of batches"
		return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


	def hasNext(self):
		"iterator"
		return self.currIdx + self.batchSize <= len(self.samples)
		
		
	def getNext(self):
		"iterator"
		batchRange = range(self.currIdx, self.currIdx + self.batchSize)
		gtTexts = [self.samples[i].gtText for i in batchRange]
		imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(gtTexts, imgs)

if __name__ == "__main__":
	l = ["shivam", "shiväm", "shivaö", "ühivam"] 
	for ll in l:
		print(change_cvl_database_chars(ll))