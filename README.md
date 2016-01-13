# FinetunedModelZoo
This repository holds GoogLeNet fine-tuned models that achieve state-of-the-art performance on various datasets. It currently includes models for:

* Bangla (Bengali) handwritten digit recognition dataset
* MNIST (Latin) handwritten digit recognition dataset
* ICDAR (Chinese) handwritten character recognition 
* CIFAR10
* CIFAR100

All pretrained models are stored in Caffe format.

Under each folder there is a bash script: classify_<dataset>.sh that can be used to evaluate one of the models against the test data.
An installed Caffe environment is required in order to run the evaluation scripts.