# FinetunedModelZoo
This repository holds GoogLeNet fine-tuned models that achieve state-of-the-art performance on various datasets. It currently includes models for:

* Bangla (Bengali) handwritten digit recognition dataset - 99.73% accuracy
* MNIST (Latin) handwritten digit recognition dataset - 99.43% accuracy
* ICDAR (Chinese) handwritten character recognition  - 95.28% accuracy
* CIFAR10 - 95.62% accuracy
* CIFAR100 - 80.66% accuracy

All pretrained models are stored in Caffe format and can be further fine-tuned for different problems.

Under each folder there is a bash script: classify_<dataset>.sh that can be used to evaluate one of the models against the test data.
An installed Caffe environment is required in order to run the evaluation scripts.
Note: In order to start the bash evaluation scripts you need to edit them and provide a test image folder as last parameter.