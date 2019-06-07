This is the read me for codes in NNeval classify folder 

==========================================================================================

test_feature_extraction.py  (py27) this is the code for extracting scores of various metrics which are used as features

Please note that the scores for SPICE and Meteor were extracted before hand and saved in a json file, due to python comptibility issues. The json files were then opened in the feature extraction module and the scores were included in the feature vector. 

All the metric implementations were taken from the MSCOCO official evaluation server, except for WMD. 
===========================================================================================

nn_classify_test_nneval.py (py35) is the test bed for the test set
===========================================================================================

nn_classify_model_nneval (py35) creates the nneval model
===========================================================================================

nn_classify_utils_nneval (py35) contains some utility functions

===========================================================================================

Please follow the following order of running the scripts

1) test_feature_extraction.py (make sure the spice and meteor scores are extracted before hand or modify the code to extract within the script)

2) nn_classify_test_nneval.py- generates the scores for captions, given the feature file. 

