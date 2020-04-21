import os
import sys
import numpy as np
from SamplePreprocessor import preprocessor as preprocess
import cv2
from DataLoader import Batch, DataLoader, FilePaths
from SamplePreprocessor import preprocessor as preprocess
from SamplePreprocessor import wer
from Model import DecoderType, Model
import tensorflow as tf
# import argparse

img_size = (128,32)

def make_batches(folder_name, images_list):
    max_images_per_batch = 60
    quo = len(images_list)//max_images_per_batch
    rem = len(images_list)%max_images_per_batch
    batch_range = range(0, max_images_per_batch)

    batches_list = []
    cnt = 0
    while(quo > 0):
        gtTexts = [None for i in range(max_images_per_batch)]
        imgs = [preprocess(cv2.imread(folder_name + "/" + images_list[i], cv2.IMREAD_GRAYSCALE), img_size) for i in range(cnt*max_images_per_batch,  (cnt+1)*max_images_per_batch)]
        batches_list.append(Batch(gtTexts, imgs))
        cnt += 1
        quo -= 1
    
    if(quo == 0 and rem > 0):
        gtTexts = [None for i in range(rem)]
        imgs = [preprocess(cv2.imread(folder_name + "/" + images_list[i], cv2.IMREAD_GRAYSCALE), img_size) for i in range(cnt*max_images_per_batch, len(images_list))]
        batches_list.append(Batch(gtTexts, imgs))
    
    return batches_list

# Requires a model definition and a batch
def infer(model, batch):
    # Reset the model defintion 
    tf.reset_default_graph()
    recognised = model.inferBatch(batch)

    return recognised

def main():
    
     # Define the DecorderType
    decorder = DecoderType.BestPath
    # Define the folder in which images are present
    folder_name = sys.argv[1]
    model = Model(open(FilePaths.fnCharList).read(), decorder, mustRestore= True)

    # Make batches list 
    batches_list = make_batches(folder_name, os.listdir(folder_name))
    print(batches_list)

    for i,batch in enumerate(batches_list):
        print("VALIDATIING BATCH {}".format(i))
        resultant_text = infer(model, batch)
        print(resultant_text)


if __name__ == "__main__":
    main()







