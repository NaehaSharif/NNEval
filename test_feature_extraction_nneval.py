
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:37:33 2018
feature extraction for test set
@author: 22161668
# this is the code for extracting scores of various metrics

13) Rogue-L score
14) Meteor score
15) WMD score
16-19) Blue 1,2,3,4 scores
20) SPICE score
21) Cider score

--variable info----

@author: 22161668 naeha sharif
"""

import os
import json 
import numpy as np
from nltk.corpus import stopwords
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from datetime import datetime
import re
import gensim.models
import math
import operator
 


data_path='D:/NNeval_classification/data'
stop_words = stopwords.words('english')
stat_path = os.path.join(data_path, "train/train_stats.json")
embeddings =gensim.models.KeyedVectors.load_word2vec_format( "GoogleNews-vectors-negative300.bin" , binary=True ) 
print( 'word to vec loaded') 

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def _parse_sentence(s): # this function parses a given string
    s = s.replace('.', '') # returns a copy of the string in which the occurrences of old have been replaced with new, s.replace(old, new)
    s = s.replace(',', '')
    s = s.replace('"', '')
    s = s.replace("'", '')
    s = s.lower() # coverts the string to lower case
    s = re.sub("\s\s+", " ", s) # \s\s+ you are looking for a space followed by 1 or more spaces in the string and trying to replace it 
    s = s.split(' ') #split or breakup a string and add the data to a string array using a defined separator.
    return s

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''   
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def dot_product2(v1, v2):
    return sum(map(operator.mul, v1, v2))

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def vector_cos(v1, v2):
    prod = dot_product2(v1, v2)
    len1 = math.sqrt(dot_product2(v1, v1))
    len2 = math.sqrt(dot_product2(v2, v2))
    return float(prod)/float(len1 * len2)

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

option=['test_pascal1']


check_flag=0 # flag to load wmd only once
 
for split in option:
    
 
    reference_path = os.path.join(data_path, "test/test_references_%s.json" %(split))
    candidate_path = os.path.join(data_path, "test/test_captions_%s.json" %(split))
    meteor_score_path = os.path.join(data_path, "test/%s_meteor_sc.json" %(split))
    spice_score_path = os.path.join(data_path, "test/%s_spice_sc.json" %(split))
    feature_path=os.path.join(data_path, "test/%s_features.json" %(split)) 
            
    stat={}
        
        # load caption data
    with open(reference_path, 'r') as f:
            ref = json.load(f)
    with open(candidate_path, 'r') as f:
            cand = json.load(f)
            
    print ('tokenization...')
    tokenizer = PTBTokenizer()
    ref  = tokenizer.tokenize( ref)
    cand = tokenizer.tokenize(cand)
    hypo = cand
    print( ' tokenzation done')
        
    assert(hypo.keys() == ref.keys())
    ImgId=hypo.keys() # for ensuring that all metrics get the same keys and return values in the same order
        
    #=====================================================================================================================    
    print( ' start of feature extraction')
    time_now = datetime.now()
    #=====================================================================================================================

    '''''''''''''''''''''''''''''''''''''1-4, 13-14'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
    # compute bleu scores#########################
    print('starting BLEU scores')
    
    blscore, blscores=Bleu(4).compute_score(ref,hypo,ImgId)
    
    print('BLEU scores done\n time:  {}'.format(datetime.now()-time_now))
    #####################################################################################################

    '''''''''''''''''''''''''''''''15'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print('--starting Rogue_L--')
    time_now = datetime.now()
    # longest Common subsequence##################
    Rogue_L,Rogue_Lscores= Rouge().compute_score(ref,hypo,ImgId)
    
    print('--Rogue_l done--\n time:  {}'.format(datetime.now()-time_now))
    #######################################################################################################    
        
    '''''''''''''''''''''''''''''''''16'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Meteor scores 
    meteor_scores=[]
    with open(meteor_score_path) as file:
        mt_file=json.load(file)
        
    for ids in ImgId:
        meteor_scores.append(mt_file[str(ids)]) # meteor scores json was saved with str as keys 
           
    print(' Meteor_done\n time:  {}'.format(datetime.now()-time_now))
    #############################################################################################
    
    '''''''''''''''''''''''''''''''''17'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print( '--wmd starting--')
    if(check_flag==0):
        wmd_model = embeddings
        wmd_model.init_sims(replace=True) # 
        check_flag=1
    
    wmd_score=[]
    wmd_score_mean=[]
    wmd_score_min=[]
    
    _count=0
    time_now = datetime.now()
    
    for id_ref in ImgId:
        c1=hypo[id_ref][0]
        c1= c1.lower().split()
        c1 = [w_ for w_ in c1 if w_ not in stop_words]
        
        distance=[]
        
        for refs in ref[id_ref]:
            c2=refs
            c2= c2.lower().split()
            c2 = [w_ for w_ in c2 if w_ not in stop_words]
            temp= wmd_model.wmdistance(c1, c2)
            
            if (np.isinf(temp)):
                temp=1000
                #print('found INF at {}, replacing with 1000'.format(id_ref))
            
            distance.append(temp)
            
        wmd_dis=min(distance)
        wmd_similarity=np.exp(-wmd_dis)
        wmd_score.append(wmd_similarity)
        time_now1 = datetime.now()            
#        if(_count%1000==0):
#            print(_count) 
#            
#        _count=_count+1 
        
    print(' wmd_done\n time:  {}'.format(datetime.now()-time_now))
    #######################################################################################################
#    '''''''''''''''''''''''''''''''18'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print('--starting Cider--')
    df_mode='coco-val-df'
    
    CIDer=Cider(df=df_mode) # using coco-val-df as tf idf'
    
    cider_all, cider_scores=CIDer.compute_score(ref,hypo,ImgId)

    print('--Cider done--\n time:  {}'.format(datetime.now()-time_now))
    
    #######################################################################################################    
    '''''''''''''''''''''''''''''''19'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
   # Spice scores 
    spice_scores=[]
    with open(spice_score_path) as file:
        sp_file=json.load(file)
        
    for ids in ImgId:
        spice_scores.append(sp_file[str(ids)]) # meteor scores json was saved with str as keys 
           
    print(' Spice_done\n time:  {}'.format(datetime.now()-time_now))
        
    #######################################################################################################
    
    #"""""""""""""""""""""""""""""""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #----------------------Sentence Embedding features -------------------------------------------------
    

    features={}
    for i,ids in enumerate(ImgId):
        
            
          features[ids]=[
                         meteor_scores[i],
                         wmd_score[i],
                         blscores[0][i],
                         blscores[1][i],
                         blscores[2][i],
                         blscores[3][i],
                         spice_scores[i],
                         cider_scores[i]/10]
     
            
            

    with open(feature_path, 'w') as f:
        json.dump(features,f)    
        
    print( '-- All {} features saved--\n time:  {}'.format(split, datetime.now()-time_now))
    
    #----------------------------------------------------------------------------------------------
     #----------------------------------------------------------------------------------------------
     #----------------------------------------------------------------------------------------------
    
 