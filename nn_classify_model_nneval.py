# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:24:35 2018

@author: 22161668
"""

import tensorflow as tf
import configuration_nneval # class that controls hyperparameters


model_config = configuration_nneval.ModelConfig()
training_config = configuration_nneval.TrainingConfig()

import random
beta=0.0001

# this code builds the nneval model.
def hidden_layers(nneval_out, config, initializer) :
    

    
    initial_fc1 = tf.Variable(((2/config.nneval_insize)**0.5)*tf.random_normal([config.nneval_insize, config.layer1out_size],seed=123))
    initial_fc2 = tf.Variable(((2/config.layer1out_size)**0.5)*tf.random_normal([config.layer1out_size, config.layer2out_size],seed=123))  

    with tf.variable_scope("layer1"):
        
        
        W1 = tf.get_variable('W1', initializer=initial_fc1.initialized_value())
        b1 = tf.get_variable('b1', [config.layer1out_size], initializer=tf.constant_initializer(0.0))
        Layer_1out = tf.matmul(nneval_out, W1) + b1 

        Layer1_out=tf.nn.relu(Layer_1out)
        
    with tf.variable_scope("layer2"):

        
        W2 = tf.get_variable('W2', initializer=initial_fc2.initialized_value())
        b2 = tf.get_variable('b2', [config.layer2out_size], initializer=tf.constant_initializer(0.0))
        Layer_2out = tf.matmul(Layer1_out, W2) + b2 
        

        Layer2_out = tf.nn.relu(Layer_2out)

        
    return Layer2_out,W1,W2,b1,b2

#---------------------------------------------------------------------------------------------   
def nn_out_layers(Layer2_out, config, initializer) :  
    
    initial_fcout = tf.Variable(((2/config.layer2out_size)**0.5)*tf.random_normal([config.layer2out_size, config.classes],seed=123)) 
#    
    with tf.variable_scope("out"):
        
        W_out = tf.get_variable('W_out', initializer=initial_fcout.initialized_value())
        b_out = tf.get_variable('b_out', [config.classes], initializer=tf.constant_initializer(0.0))
        Layer_out = tf.matmul(Layer2_out, W_out) + b_out
    
    return Layer_out,W_out,b_out 



def build_model(config):

    """Basic setup.

    Args:
      config: Object containing configuration parameters."""

    tf.set_random_seed(123)
    random.seed(123)

    initializer = tf.random_uniform_initializer(minval=-config.initializer_scale,
                                                maxval=config.initializer_scale)

     # -----placheolders for the model--------------------------------------------

    
    """Placeholders for feature inputs"""
    # An int32 Tensor with shape [batch_size, feature_size].
    sentence_features= tf.placeholder(tf.float32, [None, config.sentence_feature_size], name='image_feature')
    
    
    """Plceholder for true output"""
    # An int32 Tensor with shape [batch_size, 1].
    true_out= tf.placeholder(tf.int32, shape=[None], name='targert_score') # scores for the output
    
    diag=[1,1]
    score_embed=tf.diag(diag)
    
    target= tf.nn.embedding_lookup(score_embed,true_out)
    
 
    global_step = None
  

    """Sets up the global step Tensor."""
    global_step = tf.Variable(initial_value=0,
                              name="global_step",
                              trainable=False,
                              collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    
    
    #---------------------------------------------------------------------------------------------------

    """#---------define the model --------------------------------------------------------------------"""
    """==============================================================================================="""
   
    with tf.variable_scope("hiddenlayer") as h_layer_scope:
        
        out_hidden_p,w1,w2,b1,b2=hidden_layers(sentence_features, config, initializer) # scores for the positive image-sentence pairs
        

    
    with tf.variable_scope("out_layer") as o_pre_layer_scope:
       
        out_fine,wout_fine,b_out=nn_out_layers(out_hidden_p, config, initializer)
        

    #---------------- regularization terms ----------------------------------------------------
    
    reg1=tf.nn.l2_loss(w1)
    reg2=tf.nn.l2_loss(w2)
    reg3=tf.nn.l2_loss(wout_fine)
    

   #----------------------------------------------------------------------------------------
    nn_score=tf.nn.softmax(out_fine)
    
    """ For cross entropy loss"""
    layer_out=tf.reshape(out_fine,[-1,config.classes])
    batchloss_fine =tf.nn.softmax_cross_entropy_with_logits(logits=layer_out, labels=target) 

    #--------------------------------------------------------------------------------------------------
    # --------------define loss function-------------------------------------------------------------
   
    total_loss_fine =tf.reduce_mean(batchloss_fine)+beta*(reg1+reg2+reg3) 
    
    
    #-------------define accuracy ----------------------------------------------------------
    Yo = tf.nn.softmax(layer_out, name='Yo')        
    Y = tf.argmax(Yo, 1)                          
    
    accuracy_fine = tf.reduce_mean(tf.cast(tf.equal(tf.cast(true_out, tf.uint8), tf.cast(Y, tf.uint8)), tf.float32))
    predicted_label=tf.reshape(Y,[-1,1])
   
    #-----------------------------------------------------------------------------------------------
    """ summaries"""
    #------------define summary functions------------------------------------------------------------
    ### pre-train----------------------
    fine_loss_hist_summary=tf.summary.histogram("fine_histogram_loss", total_loss_fine)
    fine_accuarcy_hist_summary=tf.summary.histogram("fine_histogram_acc", accuracy_fine)
    weight1=tf.summary.histogram("weight_w1", w1)
    weight2=tf.summary.histogram("weight_w2", w2)
    weight3=tf.summary.histogram("weight_wout_fine", wout_fine)

    fine_loss_scalar_sumary=tf.summary.scalar("fine_scalar_loss", total_loss_fine)
    fine_accuracy_scalar_sumary=tf.summary.scalar("fine_scalar_acc", accuracy_fine)
    

    
    summaries_fine = tf.summary.merge([fine_loss_hist_summary,fine_accuarcy_hist_summary,
                                       fine_loss_scalar_sumary,fine_accuracy_scalar_sumary,
                                       weight1,weight2,weight3]) 
    
    
    #-------------------------------------------------------------------------------------------------
    Max_Results_to_save = training_config.models_to_save_per_epoch*training_config.total_num_epochs
    
    
    return dict(
            total_loss_fine = total_loss_fine,
            accuracy_fine = accuracy_fine,
            predicted_label = predicted_label,
            sentence_features= sentence_features,
            wout_fine = wout_fine,
            true_out=true_out,
            nn_score=nn_score,
            saver = tf.train.Saver(max_to_keep=Max_Results_to_save),
            global_step = global_step,
            summaries_fine=summaries_fine
        )