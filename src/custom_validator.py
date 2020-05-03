import os
import sys
import numpy as np
from SamplePreprocessor import preprocessor as preprocess
import cv2
from DataLoader import Batch, DataLoader, FilePaths
from SamplePreprocessor import preprocessor as preprocess
from SamplePreprocessor import wer
from Model import DecoderType, Model
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# warnings.filterwarnings('ignore',category=FutureWarning)
# import argparse

img_size = (128,32)
def normalise_And_center(img):
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def make_batches(folder_name, images_list):
    # Sort the images_list lexicographically
    # print(images_list)
    images_list.sort()
    # print(images_list)
    max_images_per_batch = 60
    quo = len(images_list)//max_images_per_batch
    rem = len(images_list)%max_images_per_batch
    batch_range = range(0, max_images_per_batch)

    batches_list = []
    images_filepaths = []
    cnt = 0
    while(quo > 0):
        new_images = []
        gtTexts = [None for i in range(max_images_per_batch)]
        imgs = [preprocess(cv2.imread(folder_name + "/" + images_list[i], cv2.IMREAD_GRAYSCALE), img_size) for i in range(cnt*max_images_per_batch,  (cnt+1)*max_images_per_batch)]
        batches_list.append(Batch(gtTexts, imgs))
        for i in range(cnt*max_images_per_batch,  (cnt+1)*max_images_per_batch):
            new_images.append(folder_name + "/" + images_list[i])
        cnt += 1
        quo -= 1
        images_filepaths.append(new_images)
    
    if(quo == 0 and rem > 0):
        new_images = []
        gtTexts = [None for i in range(rem)]
        imgs = [preprocess(cv2.imread(folder_name + "/" + images_list[i], cv2.IMREAD_GRAYSCALE), img_size) for i in range(cnt*max_images_per_batch, len(images_list))]
        batches_list.append(Batch(gtTexts, imgs))
        for i in range(cnt*max_images_per_batch,  len(images_list)):
            new_images.append(folder_name + "/" + images_list[i])

        images_filepaths.append(new_images)
    return batches_list, images_filepaths
    # return batches_list

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
    batches_list, images_filepaths= make_batches(folder_name, os.listdir(folder_name))
    for batchnum,batch in enumerate(batches_list):
        print("VALIDATIING BATCH {}".format(batchnum))
        resultant_text = infer(model, batch)
        for  j in range(len(images_filepaths[batchnum])):
            print(images_filepaths[batchnum][j][3:] + " : " +  resultant_text[j])


if __name__ == "__main__":
    main()







