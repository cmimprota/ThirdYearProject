import os
import sys
import time
import pickle

import random
import h5py

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg') # needed for TKinter
#import matplotlib.pyplot as plt

import tensorflow as tf

# tensorflow implementation of "A Hierarchical Approach for Generating Descriptive Image Paragraphs" (Krause et al., 2016)


os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

#############################################################
################## HRNN_features2paragraph ##################
#############################################################
#                                                           #
#   Input: features of 50 most salient regions of an image  #
#   Output: paragraph describing image                      #
#############################################################

class HRNN_features2paragraph():
    
    ################################
    ###  __init__
    #
    #   Initialise model
    
    
    def __init__(self, numberOfWords,
                       batchSize,
                       numberOfSalientRegions,
                       wordEmbeddingVector_dimension,
                       featureVector_dimension,
                       featurePooling_dimension,
                       sentenceRNN_singleLayerLSTM_hiddenSize,
                       sentenceRNN_twoLayerFullyConnectedNetwork_size,
                       wordRNN_twoLayerLSTM_hiddenSize,
                       Smax,
                       Nmax,
                       Tstop,
                       biasVector=None):

        # init
        self.numberOfWords = numberOfWords
        self.batchSize = batchSize
        self.numberOfSalientRegions = numberOfSalientRegions
        self.wordEmbeddingVector_dimension = wordEmbeddingVector_dimension

        ### Region Detection and Region Pooling ###
        self.featureVector_dimension = featureVector_dimension
        self.featurePooling_dimension = featurePooling_dimension

        self.regionPooling_projectionMatrix_W = tf.Variable(tf.random_uniform([featureVector_dimension, featurePooling_dimension], -0.1, 0.1), name='regionPooling_projectionMatrix_W')
        self.regionPooling_bias_b = tf.Variable(tf.zeros([featurePooling_dimension]), name='regionPooling_bias_b')

        #### Hierarchical Recurrent Network ###

        ## Sentence RNN ##

        # Single-Layer LSTM
        self.sentenceRNN_singleLayerLSTM_hiddenSize = sentenceRNN_singleLayerLSTM_hiddenSize
        
        self.sentenceRNN_twoLayerFullyConnectedNetwork_size = sentenceRNN_twoLayerFullyConnectedNetwork_size

        # https://stackoverflow.com/questions/41789133/what-are-c-state-and-m-state-in-tensorflow-lstm
        self.sentenceLSTM = tf.nn.rnn_cell.LSTMCell(sentenceRNN_singleLayerLSTM_hiddenSize, state_is_tuple=True)

        # Logistic Classifier
        self.logisticClassifier_W = tf.Variable(tf.random_uniform([sentenceRNN_singleLayerLSTM_hiddenSize, 2], -0.1, 0.1), name='logisticClassifier_W')
        self.logisticClassifier_b = tf.Variable(tf.zeros(2), name='logisticClassifier_b')

        # Double-Layer Fully Connected Network
        self.sentenceRNN_fullyConnected_layer1_W = tf.Variable(tf.random_uniform([sentenceRNN_singleLayerLSTM_hiddenSize, sentenceRNN_twoLayerFullyConnectedNetwork_size], -0.1, 0.1), name='sentenceRNN_fullyConnected_layer1_W')                  # 512 x 1024
        self.sentenceRNN_fullyConnected_layer1_b = tf.Variable(tf.zeros(sentenceRNN_twoLayerFullyConnectedNetwork_size), name='sentenceRNN_fullyConnected_layer1_b')                                                                                # 1024
        self.sentenceRNN_fullyConnected_layer2_W = tf.Variable(tf.random_uniform([sentenceRNN_twoLayerFullyConnectedNetwork_size, sentenceRNN_twoLayerFullyConnectedNetwork_size], -0.1, 0.1), name='sentenceRNN_fullyConnected_layer2_W')          # 1024 x 1024
        self.sentenceRNN_fullyConnected_layer2_b = tf.Variable(tf.zeros(1024), name='sentenceRNN_fullyConnected_layer2_b')
        
        ## Word RNN ##
        self.wordRNN_twoLayerLSTM_hiddenSize = wordRNN_twoLayerLSTM_hiddenSize

        # word LSTM
        # https://github.com/tensorflow/tensorflow/issues/16186
        def newWordLSTM():
            newLSTM = tf.nn.rnn_cell.LSTMCell(wordRNN_twoLayerLSTM_hiddenSize, state_is_tuple=True)
            return newLSTM
        self.wordLSTM = tf.nn.rnn_cell.MultiRNNCell([newWordLSTM() for _ in range(2)], state_is_tuple=True)

        ### startTraininging and Sampling ###
        self.Smax = Smax    # max number of sentences
        self.Tstop = Tstop  # sentence threshold
        self.Nmax = Nmax    # max number of words
         
        ### Word embedding ###
        
        #with tf.device('/gpu:0'): #needed ?
        self.wordEmbedding = tf.Variable(tf.random_uniform([numberOfWords, wordEmbeddingVector_dimension], -0.1, 0.1), name='wordEmbedding')
                                                                                                                                  
        # weights 
        self.wordEmbedding_W = tf.Variable(tf.random_uniform([wordRNN_twoLayerLSTM_hiddenSize, numberOfWords], -0.1,0.1), name='wordEmbedding_W')

        # bias
        if biasVector is not None:
            self.wordEmbedding_b = tf.Variable(biasVector.astype(np.float32), name='wordEmbedding_b')
        else:
            self.wordEmbedding_b = tf.Variable(tf.zeros([numberOfWords]), name='wordEmbedding_b')

    ################################
    ### modelConstruction
    #
    #   construct the model
    

    def modelConstruction(self):
        
        batchFeatures = tf.placeholder(tf.float32, [self.batchSize, self.numberOfSalientRegions, self.featureVector_dimension])
        reshaped_batchFeatures = tf.reshape(batchFeatures, [-1, self.featureVector_dimension])

        # TODO
        pooledFeatures = tf.matmul(reshaped_batchFeatures, self.regionPooling_projectionMatrix_W) + self.regionPooling_bias_b
        pooledFeatures = tf.reshape(pooledFeatures, [self.batchSize, 50, self.featurePooling_dimension])

        # TODO
        reduced_pooledFeatures = tf.math.reduce_max(pooledFeatures, reduction_indices=1)

        # receive the [continue:0, stop:1] lists
        # example: [0, 0, 0, 0, 1, 1], it means this paragraph has five sentences
        distribution_sentenceState = tf.placeholder(tf.int32, [self.batchSize, self.Smax])

        # TODO
        # receive the ground truth words, which has been changed to index use wordToIx function
        captions = tf.placeholder(tf.int32, [self.batchSize, self.Smax, self.Nmax+1])
        captions_masks = tf.placeholder(tf.float32, [self.batchSize, self.Smax, self.Nmax+1])

        # initialise sentence RNN state
        stateOfSentence = self.sentenceLSTM.zero_state(batch_size=self.batchSize, dtype=tf.float32)

        # initialise parameters
        probabilities = []
        loss = 0.0
        sentenceLoss = 0.0
        wordLoss = 0.0
        sentenceLambda = 5.0
        wordLambda = 1.0

        print('Start build model:')
        
        for sentence in range(0, self.Smax):
            
            if sentence > 0:
                tf.get_variable_scope().reuse_variables()

            # https://www.tensorflow.org/api_docs/python/tf/variable_scope
            with tf.variable_scope('sentenceLSTM', reuse=tf.AUTO_REUSE):
                outputSentence, stateOfSentence = self.sentenceLSTM(reduced_pooledFeatures, stateOfSentence)

            with tf.name_scope('fullyConnected_layer1'):
                layer1Output = tf.nn.relu(tf.matmul(outputSentence, self.sentenceRNN_fullyConnected_layer1_W) + self.sentenceRNN_fullyConnected_layer1_b )
            with tf.name_scope('fullyConnected_layer2'):
                vector_sentenceTopic = tf.nn.relu( tf.matmul(layer1Output, self.sentenceRNN_fullyConnected_layer2_W) + self.sentenceRNN_fullyConnected_layer2_b )

            #
            matmulbiases_sentenceRNN = tf.nn.xw_plus_b( outputSentence, self.logisticClassifier_W, self.logisticClassifier_b )
            label_sentenceRNN = tf.stack([ 1 - distribution_sentenceState[:, sentence], distribution_sentenceState[:, sentence] ])
            label_sentenceRNN = tf.transpose(label_sentenceRNN)
            
            # https://github.com/ibab/tensorflow-wavenet/issues/223
            loss_sentenceRNN = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_sentenceRNN, logits=matmulbiases_sentenceRNN)
            loss_sentenceRNN = tf.reduce_sum(loss_sentenceRNN)/self.batchSize
            loss += loss_sentenceRNN * sentenceLambda
            sentenceLoss += loss_sentenceRNN

       
            topic = tf.nn.rnn_cell.LSTMStateTuple(vector_sentenceTopic[:, 0:512], vector_sentenceTopic[:, 512:])
            wordState = (topic, topic)
            
            for word in range(0, self.Nmax):
                if word > 0:
                    tf.get_variable_scope().reuse_variables()

                #with tf.device('/gpu:0'):
                current_wordEmbedding = tf.nn.embedding_lookup(self.wordEmbedding, captions[:, sentence, word])

                with tf.variable_scope('wordLSTM', reuse=tf.AUTO_REUSE):
                    outputWord, wordState = self.wordLSTM(current_wordEmbedding, wordState)

                # http://stackoverflow.com/questions/33681517/tensorflow-one-hot-encoder
                labels = tf.reshape(captions[:, sentence, word+1], [-1, 1])
                indices = tf.reshape(tf.range(0, self.batchSize, 1), [-1, 1])
                
                # https://www.tensorflow.org/api_docs/python/tf/concat
                concated = tf.concat([indices, labels], 1)
                # TODO: update to sparse.to_dense
                labels = tf.sparse_to_dense(concated, tf.stack([self.batchSize, self.numberOfWords]), 1.0, 0.0)

                # 
                logoddsWords = tf.nn.xw_plus_b(outputWord[:], self.wordEmbedding_W, self.wordEmbedding_b)
                crossEntropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logoddsWords, labels=labels)
                crossEntropy = crossEntropy * captions_masks[:, sentence, word]
                loss_wordRNN = tf.reduce_sum(crossEntropy) / self.batchSize
                loss += loss_wordRNN * wordLambda
                wordLoss += loss_wordRNN

        return batchFeatures, distribution_sentenceState, captions, captions_masks, loss, sentenceLoss, wordLoss

    ################################
    ### modelGeneration
    #
    #   generate the model

    def modelGeneration(self):
        # 
        features = tf.placeholder(tf.float32, [1, self.numberOfSalientRegions, self.featureVector_dimension])
        # 
        reshaped_features = tf.reshape(features, [-1, self.featureVector_dimension])

        # 
        pooledFeatures = tf.matmul(reshaped_features, self.regionPooling_projectionMatrix_W) + self.regionPooling_bias_b
        pooledFeatures = tf.reshape(pooledFeatures, [1, 50, self.featurePooling_dimension])
        reduced_pooledFeatures = tf.reduce_max(pooledFeatures, reduction_indices=1)

        # initialize the sentenceLSTM state
        stateOfSentence = self.sentenceLSTM.zero_state(batch_size=1, dtype=tf.float32)

        # 
        paragraph = []

        # 
        probabilities = []

        # 
        print('Generating the model.')

        # sentence RNN
        
        for sentence in range(0, self.Smax):
            if sentence > 0:
                tf.get_variable_scope().reuse_variables()

            # 
            with tf.variable_scope('sentenceLSTM', reuse=tf.AUTO_REUSE):
                outputSentence, stateOfSentence = self.sentenceLSTM(reduced_pooledFeatures, stateOfSentence)

            # 
            with tf.name_scope('fullyConnected_layer1'):
                layer1Output = tf.nn.relu( tf.matmul(outputSentence, self.sentenceRNN_fullyConnected_layer1_W) + self.sentenceRNN_fullyConnected_layer1_b)
            with tf.name_scope('fullyConnected_layer2'):
                vector_sentenceTopic = tf.nn.relu( tf.matmul(layer1Output, self.sentenceRNN_fullyConnected_layer2_W) + self.sentenceRNN_fullyConnected_layer2_b)

            # result = x*W+b
            matmulbiases_sentenceRNN = tf.nn.xw_plus_b(outputSentence, self.logisticClassifier_W, self.logisticClassifier_b)
            probability = tf.nn.softmax(matmulbiases_sentenceRNN)
            probabilities.append(probability)

            # 
            sentences = []

            # word LSTM 
            topic = tf.nn.rnn_cell.LSTMStateTuple(vector_sentenceTopic[:, 0:512], vector_sentenceTopic[:, 512:])
            wordState = (topic, topic)
            
            # 
            
            for word in range(0, self.Nmax):
                if word > 0:
                    tf.get_variable_scope().reuse_variables()

                if word == 0:
                    #with tf.device('/gpu:0'):
                    # get word embedding of BOS (index = 0)
                    current_wordEmbedding = tf.nn.embedding_lookup(self.wordEmbedding, tf.zeros([1], dtype=tf.int64))

                with tf.variable_scope('wordLSTM', reuse=tf.AUTO_REUSE):
                    outputWord, wordState = self.wordLSTM(current_wordEmbedding, wordState)

                logoddsWords = tf.nn.xw_plus_b(outputWord, self.wordEmbedding_W, self.wordEmbedding_b)
                maxProbabilityIndex = tf.argmax(logoddsWords, 1)[0]
                sentences.append(maxProbabilityIndex)

                #with tf.device('/gpu:0'):
                current_wordEmbedding = tf.nn.embedding_lookup(self.wordEmbedding, maxProbabilityIndex)
                current_wordEmbedding = tf.expand_dims(current_wordEmbedding, 0)

            paragraph.append(sentences)

        return features, paragraph, probabilities, vector_sentenceTopic

##########################################
### preProBuildWordVocab (from NeuralTalk)
#
#   startTraining the model

def preProBuildWordVocab(sentenceIterator, wordCountThreshold=5):
    
    # Code taken from Python Reinforcement Learning Projects -  Sean Saito, Yang Wenzhuo, Rajalingappaa Shanmugamani
    # https://books.google.co.uk/books?id=VP1wDwAAQBAJ&pg=PA192&lpg=PA192&dq=ixToWord+%3D+%7B%7D+++++ixToWord%5B0%5D+%3D+%27%3Cbos%3E%27+++++ixToWord%5B1%5D+%3D+%27%3Ceos%3E%27+++++ixToWord%5B2%5D+%3D+%27%3Cpad%3E%27+++++ixToWord%5B3%5D+%3D+%27%3Cunk%3E%27++++++wordToIx+%3D+%7B%7D+++++wordToIx%5B%27%3Cbos%3E%27%5D+%3D+0+++++wordToIx%5B%27%3Ceos%3E%27%5D+%3D+1+++++wordToIx%5B%27%3Cpad%3E%27%5D+%3D+2+++++wordToIx%5B%27%3Cunk%3E%27%5D+%3D+3&source=bl&ots=1p8bV7DT8w&sig=ACfU3U3qRa46qPzqOHM2-iQTBhE7pnqbbw&hl=en&sa=X&ved=2ahUKEwi84r_0n_bgAhUugM4BHXp9DnQQ6AEwBHoECAEQAQ#v=onepage&q&f=false
    # Code taken from NeuralTalk - Andrej Karpathy
    # https://github.com/karpathy/neuraltalk/blob/master/driver.py

    print('Preprocessing word counts.')
    # count up all word counts
    wordCounter = {}
    numberOfSentences = 0

    for sentence in sentenceIterator:
        numberOfSentences += 1
        words = sentence.lower().split(' ')
        if '' in words:
            words.remove('')

        for word in words:
            # increment counter
            wordCounter[word] = wordCounter.get(word, 0) + 1

    print('Build vocabulary.')

    vocabulary = [word for word in wordCounter if wordCounter[word] >= wordCountThreshold]

    # ixToWord : map predicted indeces to words for output visualization
    ixToWord = {}
    ixToWord[0] = '<bos>'
    ixToWord[1] = '<eos>'
    ixToWord[2] = '<pad>'
    ixToWord[3] = '<unk>'

    # wordToIx : map raw words to their index in word vector matrix
    wordToIx = {}
    wordToIx['<bos>'] = 0
    wordToIx['<eos>'] = 1
    wordToIx['<pad>'] = 2
    wordToIx['<unk>'] = 3

    index = 4
    for word in vocabulary:
        wordToIx[word] = index
        ixToWord[index] = word
        index+=1
    
    np.save('./dataset/ixToWord', ixToWord)
    np.save('./dataset/wordToIx', wordToIx)

    wordCounter['<eos>'] = numberOfSentences
    wordCounter['<bos>'] = numberOfSentences
    wordCounter['<pad>'] = numberOfSentences
    wordCounter['<unk>'] = numberOfSentences


    # the bias vector is related to the log probability of the distribution of the labels (words) 
    # and how often they occur. 
    # use this vector to initialize the decoder weights 
    # so that the loss function doesn't show a huge increase in performance very quickly.

    biasVector = np.array([1.0 * wordCounter[ ixToWord[ix] ] for ix in ixToWord])
    biasVector /= np.sum(biasVector) 
    biasVector = np.log(biasVector)
    biasVector -= np.max(biasVector) 

    return wordToIx, ixToWord, biasVector


################################
### startTraining
#
#   startTraining the model

def startTraining():
    
    ## DATA PREPROCESSING ##

    # set models folder
    path_model = './models/'
    
    # get training features
    path_trainingFeatures = './dataset/training_output.h5'
    file_trainingFeatures = h5py.File(path_trainingFeatures, 'r')
    trainingFeatures = file_trainingFeatures.get('feats')

    # get training images id 
    path_trainingImages = open('./dataset/training_set.txt').read().splitlines()
    # strip ids
    trainingIDs = list(map(lambda x: os.path.basename(x).split('.')[0], path_trainingImages))

    # store indices of training ID
    indices_trainingIDs = {}
    for index, trainingID in enumerate(trainingIDs):
        indices_trainingIDs[trainingID] = index


    ## INITIALIZE THE MODEL ##
  
    with tf.variable_scope(tf.get_variable_scope()) as variableScope:
        
        model = HRNN_features2paragraph(numberOfWords = len(wordToIx),
                                        batchSize = batchSize,
                                        numberOfSalientRegions = numberOfSalientRegions,
                                        wordEmbeddingVector_dimension = wordEmbeddingVector_dimension,
                                        featureVector_dimension = featureVector_dimension,
                                        featurePooling_dimension = featurePooling_dimension,
                                        sentenceRNN_singleLayerLSTM_hiddenSize = sentenceRNN_singleLayerLSTM_hiddenSize,
                                        sentenceRNN_twoLayerFullyConnectedNetwork_size = sentenceRNN_twoLayerFullyConnectedNetwork_size,
                                        wordRNN_twoLayerLSTM_hiddenSize = wordRNN_twoLayerLSTM_hiddenSize,
                                        Smax = Smax,
                                        Nmax = Nmax,
                                        Tstop = Tstop,
                                        biasVector=biasVector)
        model_features, model_distribution_sentenceState, model_captions_matrix, model_captions_masks, totalLoss, sentenceLoss, wordLoss = model.modelConstruction()

    ## TRAIN THE MODEL ##      
    with tf.Session() as session:
        
        print("Session successfully started.")

        #
        saver = tf.train.Saver(max_to_keep=500, write_version=tf.train.SaverDef.V2)

        #
        trainOptimizer = tf.train.AdamOptimizer(learningRate).minimize(totalLoss)

        #
        tf.global_variables_initializer().run()
        
        # Initialise loss values
        # http://stackoverflow.com/questions/11874767/real-time-plotting-in-while-loop-with-matplotlib
        path_lossCurveGraphs = './no_bias_graphs'
        path_lossCurveFile = 'loss.txt'

        lossCurve = []
        wordLossCurve = []
        sentenceLossCurve = []

        file_lossCurve = open(path_lossCurveFile, 'a')
        
        #
        for epoch in range(0, numberOfEpochs):
            
            #
            epochLoss = []
            epochSentenceLoss = []
            epochWordLoss = []
            
            #
            random.shuffle( trainingIDs)

            for firstElementOfBatch, endElementOfBatch in zip(range(0, len(trainingIDs), batchSize), range(batchSize, len( trainingIDs), batchSize)):

                timeStart = time.time()

                batchIDs =  trainingIDs[firstElementOfBatch:endElementOfBatch]
                indices_batchIDs = list(map(lambda x:  indices_trainingIDs[x], batchIDs))
                batchFeatures = np.asarray(list(map(lambda x: trainingFeatures[x], indices_batchIDs)))

                batch_sentenceDistribution = np.asarray(list( map(lambda x: processedParagraphs[x][0], batchIDs)))
                batch_captions = np.asarray(list( map(lambda x: processedParagraphs[x][1], batchIDs)))
                batch_captionsMasks = np.zeros((batch_captions.shape[0], batch_captions.shape[1], batch_captions.shape[2]))
                
                # 
                nonzero_elements = np.array(list(map(lambda each_matrix: np.array(list(map(lambda x: (x != 2).sum() + 1, each_matrix ) ) ), batch_captions)))
                for batchElement in range(batchSize):
                    for index, row in enumerate(batch_captionsMasks[batchElement]):
                        row[:(nonzero_elements[batchElement, index]-1)] = 1

                #
                _, loss, lossSentence, lossWord= session.run(
                                    [trainOptimizer, totalLoss, sentenceLoss, wordLoss],
                                    feed_dict={
                                               model_features: batchFeatures,
                                               model_distribution_sentenceState: batch_sentenceDistribution,
                                               model_captions_matrix: batch_captions,
                                               model_captions_masks: batch_captionsMasks
                                    })

                # store loss values
                epochLoss.append(loss)
                epochSentenceLoss.append(lossSentence)
                epochWordLoss.append(lossWord)
                file_lossCurve.write('Epoch:' + str(epoch) + ', loss:' + str(loss) + ', sentence loss: ' + str(sentenceLoss) + ', word loss:' + str(wordLoss))

                # print progress
                elapsedTime = time.time() - timeStart
                print 'Batch element: ', firstElementOfBatch, ' epoch: ', epoch, ', loss: ', loss, ', sentence loss: ', lossSentence, ', word loss: ', lossWord, ', elapsed time: ', str(elapsedTime)
            
            lossCurve.append(np.mean(epochLoss))
            sentenceLossCurve.append(np.mean(epochSentenceLoss))
            wordLossCurve.append(np.mean(epochWordLoss))
            
            # Save model and relative graphs every 5 epochs
            if np.mod(epoch, 5) == 0:
                
                # Save model
                saver.save(session, os.path.join(path_model, 'epoch'), global_step=epoch)
                print("Saved epoch ", epoch, ".")

                # Save images
                path_imageLoss = str(epoch) + '.png'
                plt.plot(range(len(lossCurve)), lossCurve, color='r')
                plt.grid(True)
                plt.savefig(os.path.join(path_lossCurveGraphs, path_imageLoss))

                # https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
                plt.clf()
                plt.cla()

                path_imageSentenceLoss = str(epoch) + '-sentence.png'
                plt.plot(range(len(sentenceLossCurve)), sentenceLossCurve, color='g')
                plt.grid(True)
                plt.savefig(os.path.join(path_lossCurveGraphs, path_imageSentenceLoss))

                # https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
                plt.clf()
                plt.cla()

                path_imageWordLoss = str(epoch) + '-word.png'
                plt.plot(range(len(wordLossCurve)), wordLossCurve, color='b')
                plt.grid(True)
                plt.savefig(os.path.join(path_lossCurveGraphs, path_imageWordLoss))

        file_lossCurve.close()



#######################################
### Setting parameters of the model ###
#######################################

# Number of elements per batch
batchSize = 32  #TODO: try with 32,50,64
# Number of regions detected in each image
numberOfSalientRegions = 50 
#
featureVector_dimension = 4096 # feature dimensions of each regions
# 
featurePooling_dimension = 1024 # project the features to one vector, which is 1024 dimensions
 
# sentence RNN dimensions
sentenceRNN_singleLayerLSTM_hiddenSize = 512 # the sentence LSTM hidden units
sentenceRNN_twoLayerFullyConnectedNetwork_size = 1024 # the fully connected units

# Word RNN dimensions
wordRNN_twoLayerLSTM_hiddenSize = 512 # the word LSTM hidden units
wordEmbeddingVector_dimension = 1024 # the learned embedding vectors for the words

# Maximum number of sentences in paragraph
Smax = 6
# Sentence threshold 
Tstop = 0.5
# Maximum number of words in sentence
Nmax = 30


# Number of epochs
numberOfEpochs = 501
# Learning rate
learningRate = 0.0001

#######################################
###### Preprocess word embedding ######
#######################################

paragraphs = pickle.load(open('./dataset/training_groundtruth', 'rb'))
sentences = []

for _, paragraph in paragraphs.items():
    for sentence in paragraph[1]:
        # the comma is treated as a word
        sentence.replace(',', ' ,')
        sentences.append(sentence)

wordToIx, ixToWord, biasVector = preProBuildWordVocab(sentences, wordCountThreshold=2)

processedParagraphs = {}

#
for imageID, paragraph in paragraphs.items():
    textParagraph = paragraph[1]

    # remove [""] elements
    if '' in textParagraph:
        textParagraph.remove('')

    # remove [" "] elements
    if ' ' in textParagraph:
        textParagraph.remove(' ')

    # reduce number of sentences
    numberOfSentences = paragraph[0]
    if numberOfSentences > Smax:
        numberOfSentences = Smax
    # [0 , 0 , 0 , 0 , 0 , 0]
    distribution_numberOfSentences = np.zeros([Smax], dtype=np.int32)
    # i.e. if paragraph has 3 sentences then [0 , 0 , 1 , 1 , 1 , 1] --> {CONTINUE = 0, STOP = 1}
    distribution_numberOfSentences[numberOfSentences-1:] = 1

    # we multiply the number 2, because the <pad> is encoded into 2 TODO
    captions = np.ones([Smax, Nmax+1], dtype=np.int32) * 2 
    
    for indexOfSentence, sentence in enumerate(textParagraph):
        
        # the number of sentences is img_num_sents
        if indexOfSentence == numberOfSentences:
            break
        
        # treat " ," as a word 
        sentence = sentence.replace(',', ' ,')
    
        if len(sentence) > 1:
            
            # remove blank space at the beginning of the sentence
            if sentence[0] == ' ' and sentence[1] != ' ':
                sentence = sentence[1:]
            
            # remove blank space at the end of the sentence
            if sentence[-1] == ' ':
                sentence = sentence[0:-1]
            
            # remove punctuation at the end of the sentence
            if sentence[-1] == '.':
                sentence = sentence[0:-1]
        

        # add <bos> and <eos> in each sentences
        sentence = '<bos> ' + sentence + ' <eos>'

        # translate each word in a sentence into the unique number in wordToIx dict
        # when we meet the word which is not in the wordToIx dict, we use the mark: <unk>
        for indexOfWord, word in enumerate(sentence.lower().split(' ')):
            # because the biggest number of words in a sentence is Nmax, 
            if indexOfWord == Nmax:
                break
            # 
            if word in wordToIx:
                captions[indexOfSentence, indexOfWord] = wordToIx[word]
            # the word is not present in the dict
            else:
                captions[indexOfSentence, indexOfWord] = wordToIx['<unk>']

    # 
    processedParagraphs[str(imageID)] = [distribution_numberOfSentences, captions]

with open('./dataset/training_groundtruth_processed', 'wb') as file:
    pickle.dump(processedParagraphs, file)

#######################################
########### Testing Methods ###########
#######################################

################################
### fullTesting
#
# test the whole testing dataset


def fullTesting():
    testStart = time.time()
    # change the model path according to your environment
    path_model = './no_bias_models/epoch-90'

    # It's very important to use Pandas to Series this ixToWord dict
    # After this operation, we can use list to extract the word at the same time
    ixToWord = pd.Series(np.load('./dataset/ixToWord.npy').tolist())

    path_testingFeatures = './dataset/training_output.h5'
    file_testingFeatures = h5py.File(path_testingFeatures, 'r')
    testingFeatures = file_testingFeatures.get('feats')

    path_testingIDs = open('./dataset/training_set.txt').read().splitlines()
    testingIDs = map(lambda x: os.path.basename(x).split('.')[0], path_testingIDs)

    # n_words, batch_size, num_boxes, feats_dim, project_dim, sentRNN_lstm_dim, sentRNN_FC_dim, wordRNN_lstm_dim, S_max, N_max
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        model = HRNN_features2paragraph(numberOfWords = len(wordToIx),
                                        batchSize = batchSize,
                                        numberOfSalientRegions = numberOfSalientRegions,
                                        wordEmbeddingVector_dimension = wordEmbeddingVector_dimension,
                                        featureVector_dimension = featureVector_dimension,
                                        featurePooling_dimension = featurePooling_dimension,
                                        sentenceRNN_singleLayerLSTM_hiddenSize = sentenceRNN_singleLayerLSTM_hiddenSize,
                                        sentenceRNN_twoLayerFullyConnectedNetwork_size = sentenceRNN_twoLayerFullyConnectedNetwork_size,
                                        wordRNN_twoLayerLSTM_hiddenSize = wordRNN_twoLayerLSTM_hiddenSize,
                                        Smax = Smax,
                                        Nmax = Nmax,
                                        Tstop = Tstop,
                                        biasVector = biasVector)

        
        model_features, model_paragraph, model_probabilities, model_vector_sentenceTopic = model.modelGeneration()

    #sess = tf.InteractiveSession()
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, path_model)

        file_testingResults = open('training_nobias_90.txt', 'w')

        for index, testingID in enumerate(testingIDs):
            #TODO:insert progress bar
            file_testingResults.write(testingID + ':\n')

            newParagraph = []
            textParagraph = ""

            singleTestingFeature = testingFeatures[index]
            singleTestingFeature = np.reshape(singleTestingFeature, [1, 50, 4096])

            indices_paragraph, probabilities, _ = session.run(
                                                            [model_paragraph, model_probabilities, model_vector_sentenceTopic],
                                                            feed_dict={
                                                                model_features: singleTestingFeature
                                                            })

            #generated_paragraph = ixToWord[generated_paragraph_indexes]
            for indexOfSentence in indices_paragraph:
                sentence = []
                for indexOfWord in indexOfSentence:
                    sentence.append(ixToWord[indexOfWord])
                newParagraph.append(sentence)

            for index, sentence in enumerate(newParagraph):
                #
                if probabilities[index][0][0] <= Tstop:
                    break
                newSentence = ''
                firstWord=1
                for word in sentence:
                    if (firstWord==1):
                        word = word.capitalize()
                        firstWord=0
                    newSentence += word + ' '
                newSentence = newSentence.replace('<eos> ', '')
                newSentence = newSentence.replace('<pad> ', '')
                newSentence = newSentence + '.'
                newSentence = newSentence.replace(' .', '.')
                newSentence = newSentence.replace(' ,', ',')
                textParagraph +=newSentence
                if index != len(newParagraph) - 1:
                    textParagraph += ' '
            file_testingResults.write(textParagraph.encode('utf8') + '\n')
        file_testingResults.close()
    session.close()
    # https://stackoverflow.com/questions/42706761/closing-session-in-tensorflow-doesnt-reset-graph
    tf.reset_default_graph()

def hardcodedSingleTest():
    testStart = time.time()
    # change the model path according to your environment
    path_model = './no_bias_models/epoch-10'

    # It's very important to use Pandas to Series this ixToWord dict
    # After this operation, we can use list to extract the word at the same time
    ixToWord = pd.Series(np.load('./dataset/ixToWord.npy').tolist())

    path_trainingFeatures = './densecap_features/84.h5'
    file_testingFeatures = h5py.File(path_trainingFeatures, 'r')
    testingFeatures = file_testingFeatures.get('feats')

    # n_words, batch_size, num_boxes, feats_dim, project_dim, sentRNN_lstm_dim, sentRNN_FC_dim, wordRNN_lstm_dim, S_max, N_max
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        model = HRNN_features2paragraph(numberOfWords = len(wordToIx),
                                        batchSize = batchSize,
                                        numberOfSalientRegions = numberOfSalientRegions,
                                        wordEmbeddingVector_dimension = wordEmbeddingVector_dimension,
                                        featureVector_dimension = featureVector_dimension,
                                        featurePooling_dimension = featurePooling_dimension,
                                        sentenceRNN_singleLayerLSTM_hiddenSize = sentenceRNN_singleLayerLSTM_hiddenSize,
                                        sentenceRNN_twoLayerFullyConnectedNetwork_size = sentenceRNN_twoLayerFullyConnectedNetwork_size,
                                        wordRNN_twoLayerLSTM_hiddenSize = wordRNN_twoLayerLSTM_hiddenSize,
                                        Smax = Smax,
                                        Nmax = Nmax,
                                        Tstop = Tstop,
                                        biasVector = biasVector)

        
        model_features, model_paragraph, model_probabilities, model_vector_sentenceTopic = model.modelGeneration()

    #sess = tf.InteractiveSession()
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, path_model)

        newParagraph = []
        textParagraph = ""

        singleTestingFeature = np.reshape(testingFeatures, [1, 50, 4096])

        indices_paragraph, probabilities, _ = session.run(
                                                            [model_paragraph, model_probabilities, model_vector_sentenceTopic],
                                                            feed_dict={
                                                                model_features: singleTestingFeature
                                                            })

        #generated_paragraph = ixToWord[generated_paragraph_indexes]
        for indexOfSentence in indices_paragraph:
            sentence = []
            for indexOfWord in indexOfSentence:
                sentence.append(ixToWord[indexOfWord])
            newParagraph.append(sentence)

        for index, sentence in enumerate(newParagraph):
            #
            if probabilities[index][0][0] <= Tstop:
                break
            newSentence = ''
            firstWord = 1;
            for word in sentence:
                if (firstWord==1):
                    word = word.capitalize()
                    firstWord = 0
                newSentence += word + ' '
                
            newSentence = newSentence.replace('<eos> ', '')
            newSentence = newSentence.replace('<pad> ', '')
            newSentence = newSentence + '.'
            newSentence = newSentence.replace(' .', '.')
            newSentence = newSentence.replace(' ,', ',')
            textParagraph +=newSentence
            if index != len(newParagraph) - 1:
                textParagraph += ' '

        print("----------------------------------------------------------------------")
        print(textParagraph)
        print("----------------------------------------------------------------------")
        print("Time cost: " + str(time.time()-testStart))
        
    session.close()
    # https://stackoverflow.com/questions/42706761/closing-session-in-tensorflow-doesnt-reset-graph
    tf.reset_default_graph()

def singleTest(model, path):
    testStart = time.time()
    # change the model path according to your environment
    path_model = './no_bias_models/' + model

    # It's very important to use Pandas to Series this ixToWord dict
    # After this operation, we can use list to extract the word at the same time
    ixToWord = pd.Series(np.load('./dataset/ixToWord.npy').tolist())

    file_testingFeatures = h5py.File(path, 'r')
    testingFeatures = file_testingFeatures.get('feats')
    

    # n_words, batch_size, num_boxes, feats_dim, project_dim, sentRNN_lstm_dim, sentRNN_FC_dim, wordRNN_lstm_dim, S_max, N_max
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        model = HRNN_features2paragraph(numberOfWords = len(wordToIx),
                                        batchSize = batchSize,
                                        numberOfSalientRegions = numberOfSalientRegions,
                                        wordEmbeddingVector_dimension = wordEmbeddingVector_dimension,
                                        featureVector_dimension = featureVector_dimension,
                                        featurePooling_dimension = featurePooling_dimension,
                                        sentenceRNN_singleLayerLSTM_hiddenSize = sentenceRNN_singleLayerLSTM_hiddenSize,
                                        sentenceRNN_twoLayerFullyConnectedNetwork_size = sentenceRNN_twoLayerFullyConnectedNetwork_size,
                                        wordRNN_twoLayerLSTM_hiddenSize = wordRNN_twoLayerLSTM_hiddenSize,
                                        Smax = Smax,
                                        Nmax = Nmax,
                                        Tstop = Tstop,
                                        biasVector = biasVector)

        
        model_features, model_paragraph, model_probabilities, model_vector_sentenceTopic = model.modelGeneration()

    #sess = tf.InteractiveSession()
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, path_model)

        newParagraph = []
        textParagraph = ""

        singleTestingFeature = np.reshape(testingFeatures, [1, 50, 4096])

        indices_paragraph, probabilities, _ = session.run(
                                                            [model_paragraph, model_probabilities, model_vector_sentenceTopic],
                                                            feed_dict={
                                                                model_features: singleTestingFeature
                                                            })

        #generated_paragraph = ixToWord[generated_paragraph_indexes]
        for indexOfSentence in indices_paragraph:
            sentence = []
            for indexOfWord in indexOfSentence:
                sentence.append(ixToWord[indexOfWord])
            newParagraph.append(sentence)

        for index, sentence in enumerate(newParagraph):
            #
            if probabilities[index][0][0] <= Tstop:
                break
            newSentence = ''
            firstWord = 1;
            for word in sentence:
                if (firstWord==1):
                    word = word.capitalize()
                    firstWord = 0
                newSentence += word + ' '
                
            newSentence = newSentence.replace('<eos> ', '')
            newSentence = newSentence.replace('<pad> ', '')
            newSentence = newSentence + '.'
            newSentence = newSentence.replace(' .', '.')
            newSentence = newSentence.replace(' ,', ',')
            textParagraph +=newSentence
            if index != len(newParagraph) - 1:
                textParagraph += ' '

        print("----------------------------------------------------------------------")
        print(textParagraph)
        print("----------------------------------------------------------------------")
        print("Time cost: " + str(time.time()-testStart))

    session.close()
    # https://stackoverflow.com/questions/42706761/closing-session-in-tensorflow-doesnt-reset-graph
    tf.reset_default_graph()

#######################################
############# GUI Methods #############
#######################################

def loadModel(bias, model):
    path_model = "./"  + bias +"_models/" + model

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        model = HRNN_features2paragraph(numberOfWords = len(wordToIx),
                                        batchSize = batchSize,
                                        numberOfSalientRegions = numberOfSalientRegions,
                                        wordEmbeddingVector_dimension = wordEmbeddingVector_dimension,
                                        featureVector_dimension = featureVector_dimension,
                                        featurePooling_dimension = featurePooling_dimension,
                                        sentenceRNN_singleLayerLSTM_hiddenSize = sentenceRNN_singleLayerLSTM_hiddenSize,
                                        sentenceRNN_twoLayerFullyConnectedNetwork_size = sentenceRNN_twoLayerFullyConnectedNetwork_size,
                                        wordRNN_twoLayerLSTM_hiddenSize = wordRNN_twoLayerLSTM_hiddenSize,
                                        Smax = Smax,
                                        Nmax = Nmax,
                                        Tstop = Tstop,
                                        biasVector = biasVector)

        
        model_features, model_paragraph, model_probabilities, model_vector_sentenceTopic = model.modelGeneration()

    # https://stackoverflow.com/questions/37568980/tensorflow-cifar10-eval-py-errorruntimeerror-attempted-to-use-a-closed-session
    globalSession = tf.Session()
    saver = tf.train.Saver()
    saver.restore(globalSession, path_model)
    
    print 'Model ', path_model,' loaded.'
    print('Tensorflow session started.')

    return globalSession, model_features, model_paragraph, model_probabilities, model_vector_sentenceTopic

def singleTesting(path, testingID, globalSession, model_features, model_paragraph, model_probabilities, model_vector_sentenceTopic):
    
    # It's very important to use Pandas to Series this ixToWord dict
    # After this operation, we can use list to extract the word at the same time
    ixToWord = pd.Series(np.load('./dataset/ixToWord.npy').tolist())

    file_testingFeatures = h5py.File(path, 'r')
    testingFeatures = file_testingFeatures.get('feats')
    

    newParagraph = []
    textParagraph = ""
    
    singleTestingFeature = np.reshape(testingFeatures, [1, 50, 4096])

    indices_paragraph, probabilities, _ = globalSession.run(
                                                            [model_paragraph, model_probabilities, model_vector_sentenceTopic],
                                                            feed_dict={
                                                                model_features: singleTestingFeature
                                                            })

        
    for indexOfSentence in indices_paragraph:
        sentence = []
        for indexOfWord in indexOfSentence:
            sentence.append(ixToWord[indexOfWord])
        newParagraph.append(sentence)

    for index, sentence in enumerate(newParagraph):
        #
        if probabilities[index][0][0] <= Tstop:
            break
        newSentence = ''
        firstWord = 1;
        for word in sentence:
            if (firstWord==1):
                word = word.capitalize()
                firstWord = 0
            newSentence += word + ' '
                
        newSentence = newSentence.replace('<eos> ', '')
        newSentence = newSentence.replace('<pad> ', '')
        newSentence = newSentence + '.'
        newSentence = newSentence.replace(' .', '.')
        newSentence = newSentence.replace(' ,', ',')
        textParagraph +=newSentence
        if index != len(newParagraph) - 1:
            textParagraph += ' '
   
    return textParagraph

def unloadModel(session):
    session.close()
    # https://stackoverflow.com/questions/42706761/closing-session-in-tensorflow-doesnt-reset-graph
    tf.reset_default_graph()
    print('Model unloaded.')

def printGraphToTensorboard():
    
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        model = HRNN_features2paragraph(numberOfWords = len(wordToIx),
                                        batchSize = batchSize,
                                        numberOfSalientRegions = numberOfSalientRegions,
                                        wordEmbeddingVector_dimension = wordEmbeddingVector_dimension,
                                        featureVector_dimension = featureVector_dimension,
                                        featurePooling_dimension = featurePooling_dimension,
                                        sentenceRNN_singleLayerLSTM_hiddenSize = sentenceRNN_singleLayerLSTM_hiddenSize,
                                        sentenceRNN_twoLayerFullyConnectedNetwork_size = sentenceRNN_twoLayerFullyConnectedNetwork_size,
                                        wordRNN_twoLayerLSTM_hiddenSize = wordRNN_twoLayerLSTM_hiddenSize,
                                        Smax = Smax,
                                        Nmax = Nmax,
                                        Tstop = Tstop,
                                        biasVector = biasVector)

        
        #model_features, model_paragraph, model_probabilities, model_vector_sentenceTopic = model.modelGeneration()
    with tf.Session() as session:
        writer = tf.summary.FileWriter("output2", session.graph)
        writer.close()
    session.close()
