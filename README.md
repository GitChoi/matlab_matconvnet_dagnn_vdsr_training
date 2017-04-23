# Matconvnet Training Code for VDSR
A DagNN Matconvnet training implementation of "Accurate Image Super-Resolution Using Very Deep Convolutional Networks," CVPR, 2016.

## Coding Environment
* Matconvnet 1.0-beta23 from http://www.vlfeat.org/matconvnet/ installed and compiled.
* Matlab R2016a.
* Titan X Pascal.

## Files
* matconvnet-1.0-beta23: Copy this folder to your installed Matconvnet path. Includes a pdist layer function for DagNN. 
* cnn_train_dag_hardclip.m: A cnn_train_dag variant specifically for VDSR's gradient clipping.
* create_traindata.m: Creates an imdb file for training dataset.
* exe_test.m: A function for testing test images.
* extract_subim.m: A subfunction for extracting subimages from images. Used in the create_traindata.m file.
* main_train.m: A main code for training VDSR. Calls vdsr_setup.m and exe_test.m.
* vdsr_setup.m: A function for setting network structure for VDSR. Calls cnn_train_dag_hardclip.m
