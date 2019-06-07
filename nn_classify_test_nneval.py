# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:45:06 2018

@author: 22161668
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:28:23 2018
testing nneval_classify and correlation evaluation
@author: 22161668
"""

# -*- coding: utf-8 -*-

""" generate scores of the test data""" # classification
""" funcions"""
""" load_compare_data"""  # returns captions and features of the whole test data
""" _step_test"""  # generates scores for the input captions and feature pairs
""" save_result""" # saves the generated scores into a csv file



import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import scipy.stats
import math

import configuration # class that controls hyperparameters
from nn_classify_model import* # makes the main graph # its the backbone 
from nn_classify_utils import load_test_data

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

verbose = True
f=math.ceil((training_config.total_examples/model_config.train_batch_size)/training_config.models_to_save_per_epoch)
number=training_config.models_to_save_per_epoch*f*training_config.total_num_epochs
directory='D:/NNeval_classification/nnclassify_session_'+model_config.case+'/'
modelname='nn_classify_checkpoint{}.ckpt'.format(number)



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def _step_test(sess, model, features):
    
 
    nn_score= sess.run([model['nn_score']], 
                       feed_dict={model['sentence_features']: features})

    
    return nn_score   
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def process_scores(sc,BATCH_SIZE_INFERENCE):

    score_list=[]
    
    
    for i in range(BATCH_SIZE_INFERENCE):
        score_list.append(sc[0][i][1])
    
    return score_list
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 


base_dir = 'D:/NNeval_classification/data'
test=os.path.join(base_dir,'valcomp_full/valcomp_full_features.json')
features_dict, Ids = load_test_data(test)
features=features_dict['test_features'][:]
TOTAL_INFERENCE_STEP = 1
BATCH_SIZE_INFERENCE = len(features)


#----------------------------
print('Total test examples :{}'.format(BATCH_SIZE_INFERENCE))



g = tf.Graph()
with g.as_default():
    # Build the model.

    model = build_model(model_config)
    
    print('graph loaded!')
    
    # run training 
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    
    checking=[1110] # model number
    
    with tf.device('/gpu:1'): 
    
        sess.run(init)
        for s in checking:
            
            model['saver'].restore(sess, os.path.join(directory,'nn_classify_checkpoint{}.ckpt'.format(s)))
              
            
            for i in range(TOTAL_INFERENCE_STEP):
               
                sc = _step_test(sess,model,features) 
                
#            print("processing_successful")
            scores=process_scores(sc, BATCH_SIZE_INFERENCE)

            
            print("inference_successful")
 
        
        sess.close()
    
            
            