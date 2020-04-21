from __future__ import division, print_function 
import tensorflow as tf
from decorder  import Decorders as DecorderType
from decorder import FilePaths 
import codecs

# Creates the Ctc loss and decorder
# Input : BatchSize X 32 *  len(charlist)
# Input  : B X T XC 
def ctc_loss(inputs, gtTexts, seqLen, decorderType):

    # BXTXC  --> TXBXC
    ctcIn3dTBC = tf.transpose(inputs, [1, 0, 2])

   # Ground truth text as sparse tensor
    with tf.name_scope('CTC_Loss'):
        gTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[
                                        None, 2]), tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
        # Calculate loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=gtTexts, inputs= ctcIn3dTBC, sequence_length=seqLen,
                            ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True))
        
    # Lets create the decorder
    with tf.name_scope('CTC_Decorder'):
        # Best Path or WordBeamSeach supported 

        if decorderType == DecorderType.WordBeamSeach:
            # Import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            word_beam_search_module = tf.load_op_library(
                './TFWordBeamSearch.so')

            # Prepare: dictionary, characters in dataset, characters forming words
            chars = codecs.open(FilePaths.wordCharList.txt, 'r').read()
            wordChars = codecs.open(
                FilePaths.fnWordCharList, 'r').read()
            corpus = codecs.open(FilePaths.corpus.txt, 'r').read()

            # # Decoder using the "NGramsForecastAndSample": restrict number of (possible) next words to at most 20 words: O(W) mode of word beam search
            # decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 25, 'NGramsForecastAndSample', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

            # Decoder using the "Words": only use dictionary, no scoring: O(1) mode of word beam search
            decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(
                ctcIn3dTBC, dim=2), 25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))


        if decorderType == DecorderType.BestPath:
            decorder = tf.nn.ctc_greedy_decorder(
                inputs = ctcIn3dTBC, sequence_length = seqLen
            )


    # Return the loss and decorder
    return ctc_loss, decorder


