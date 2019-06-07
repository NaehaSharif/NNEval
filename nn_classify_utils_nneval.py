# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:29:54 2018
utils for nn_classify 
@author: 22161668
"""

import json
import numpy as np


def sample_minibatch(data, batch_size, split,k):
  

  labels = data['%s_labels' % split][k:k+batch_size]
  sentence_features = data['%s_features' % split][k:k+batch_size]
  

  return sentence_features, labels


def load_test_data(test):
    
  data = {}
  
  train_feature_file =test
  
  
  with open(train_feature_file) as f:  # loading training features
    test_feats=json.load(f)
 
      
  testIds=test_feats.keys()

  test_features=[]
  ids=[]
    
  for i in testIds:
       
      test_features.append(test_feats[i])
       
      ids.append(i)
            
  data['test_features']=np.array(test_features)
  

  return data,ids


#def sample_test_minibatch(data, batch_size=10, split='train',k):
#  
##  split_size = data['%s_features' % split].shape[0]
##      #mask = np.random.choice(split_size, batch_size)
##  mask = np.random.choice(split_size, batch_size)
##  mask = random.sample(range(split_size), batch_size)
#  labels = data['%s_labels' % split][k:k+batch_size]
#  #image_idxs = data['%s_imageid' % split][mask]
#  sentence_features = data['%s_features' % split][k:k+batch_size]
#  
#
#  return sentence_features, labels

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def shuffle_data(data,shuffle):
    
#    sh_data={}
#    n = len(data['train_features'])
#    shuffle=random.sample(range(0, n), n)
    
    data['train_features']=data['train_features'][shuffle]
    data['val_features']=data['val_features']
       
    data['train_labels']=data['train_labels'][shuffle]
    data['val_labels']=data['val_labels']
       
    data['train_imageid']=data['train_imageid'][shuffle]
    data['val_imageid']=data['val_imageid']
#      
    data['val_judgements']=data['val_judgements']
    #print('Shuffle data called at {}:{}'.format(((datetime.now() - time_now).seconds),(datetime.now() - time_now).seconds/60))
            
    return data


