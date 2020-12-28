
This repo consists of the fixed-point analysis code for ECE498 DL in Hardware at UIUC

The code will generate bit-precisions and dynamic ranges of activations and weights for any trained DL networks through method presented [here](http://sakr2.web.engr.illinois.edu/papers/2017/icml_draft_full.pdf). The results are saved in .npy format. 

The "compute_precision_offsets" function generates the resultant npy files. To test the accuracy of a quantized network, one should run "qfeedforward" function





