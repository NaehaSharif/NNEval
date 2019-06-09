# NNEval
Neural Network based Evaluation Metric for Image Captioning
================================================================

NNEval is a learning-based metric to evaluate image captions. Our proposed framework incorporate both lexical and semantic information into a single learned metric. This results in an evaluator that takes into account various linguistic features to assess the caption quality.

Dependencies
==========
These scripts were written in Python, and following is needed to run these scripts:

-Python 2.7 (for the tokenizer and some of the features)

-Python 3.5 (for NNEval model and metric)

-TensorFlow 1.5.0 (GPU)


Scripts
========

Here is a brief description of scripts used for NNeval:

1)test_feature_extraction_nneval.py  (py27) this is the code for extracting scores of various metrics which are used as features

=======================================================================

All the metric implementations were taken from the MSCOCO official evaluation server, except for WMD. 

=======================================================================

2)nn_classify_test_nneval.py (py35) is the test bed for the test set

=======================================================================

3)nn_classify_model_nneval (py35) creates the nneval model

=======================================================================

4)nn_classify_utils_nneval (py35) contains some utility functions

=======================================================================

*Please note that the scores for SPICE and METEOR were extracted before hand and saved in a json file, due to python comptibility issues. The json files were then opened in the feature extraction module and the scores were included in the feature vector. 

-------------------------------


Please follow the following order of running the scripts:

1) test_feature_extraction.py (make sure the spice and meteor scores are extracted before hand or modify the code to extract within the script)

2) nn_classify_test_nneval.py- generates the scores for captions, given the feature file. 


Reference
====================

Please acknowledge the following authors, if you use this code as a part of any published research:

NNEval: Neural Network based Evaluation Metric for Image Captioning
@inproceedings{sharif2018nneval,
  title={NNEval: Neural Network based Evaluation Metric for Image Captioning},
  author={Sharif, Naeha and White, Lyndon and Bennamoun, Mohammed and Afaq Ali Shah, Syed},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={37--53},
  year={2018}
}
