Full pipeline is given in the shell script **run.pipeline.sh**.

You can pass pass arguments to this bash script defining which one of the steps (preprocessing: **p**, feature representation: **f**, MAP scores: **m**, kaggle results: **k**, yaml filepath: **y**) you want to perform.

This helps you to avoid rewriting the bash script whenever there are intermediate steps that you don't want to repeat.
Here we also show you how to keep all your parameters in a **yaml file**. It helps to keep track of different parameter configurations that you may try. However, you do not have to keep your parameters in a yaml file. You can change this code as you want.

Here is an example of how to execute the script: 

    bash run.pipeline.sh -p true -f true -m true -k true -y filepath
    
As you already have functions to train kmeans and SVMs, we did not include those skeletons here.
The main TODOs will be to write the function for SURF feature extraction and for CNN feature extraction. **You can reuse your code from HW1 for kmeans and SVM training.**

Results:
#####################################
#       MED with MFCC Features      #
#####################################
=========  Event P001  =========
Evaluating the average precision (AP)
Average precision:  0.1835263835263835
=========  Event P002  =========
Evaluating the average precision (AP)
Average precision:  0.4030178905178905
=========  Event P003  =========
Evaluating the average precision (AP)
Average precision:  0.07496960528093448

#####################################
#       MED with ASR Features       #
#####################################
=========  Event P001  =========
Evaluating the average precision (AP)
Average precision:  0.9759432234432235
=========  Event P002  =========
Evaluating the average precision (AP)
Average precision:  1.0
=========  Event P003  =========
Evaluating the average precision (AP)
Average precision:  0.9999999999999998

