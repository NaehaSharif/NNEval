# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:23:03 2018

configuration for nneval classify

@author: 22161668
"""
# classification configuration


"""NNeval model and training configurations."""

class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""

    self.train_batch_size = 6650
    self.val_batch_size = 3976 # size of the validation set 

    # Scale used to initialize model variables.
    self.initializer_scale = 0.1 # we can also initialise using random variables

    self.sentence_feature_size = 8    
    self.classes = 2   
    self.nneval_insize= 8
    self.layer1out_size = 72 
    self.layer2out_size = 72 
    self.case='nneval'

class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""

# num of examples 137616 where reduced by extracting hodosh examples out
    self.total_examples =132984

    # Optimizer for training the model.
    self.optimizer = "Adam"

    self.learning_rate_decay_factor = 0.5 
    self.num_epochs_per_decay =3 

#    self.clip_gradients = 5.0
    self.models_to_save_per_epoch =2
    self.total_num_epochs = 700
