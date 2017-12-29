# My Final Year Project - A Comparison of Facial Recognition Implementations

This is a refactor of my existing work to encapsulate everything into one modular project.
The goal of this project is compare deep learning techniques against more traditional approaches
in the task of facial recognition. I will be comparing performance and efficiency in two key areas:
1) Quality of Facial Embeddings
2) Quality of Classifications

For 1), I will be comparing FaceNet embeddings against HOG feature vectors from OpenCV.
For 2), I will be comparing a Neural classifier against an SVM

These files will allow you to choose a form of getting embeddings, choose a classifier and it will then
run and measure the performance.

This is still a work in progress.

## Implemented Features
1) Dataset Parsing
2) Obtaining HOG Feature Vectors
3) SVM Classifier
4) Obtaining FaceNet Embeddings
5) K Nearest Neighbours Classifier

## Features still to be Implemented
1) Neural Classifier
2) Dataset Facial Alignment
3) Improved Performance Measuring

## Prerequisites
This requires Python 3.X

A working Tensorflow environment is required. Install instructions can be found [here](https://www.tensorflow.org/install/install_linux).

Python requirements can be found and installed from requirements.txt
`pip3 install -r requirements.txt`

Tested using the [LFW Dataset](http://vis-www.cs.umass.edu/lfw/). You can download this [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz).

You will require the FaceNet code for the FaceNet embedding part of this project.
You can download this by executing:
`git clone https://github.com/davidsandberg/facenet`
Following that you will need to add it to your python path.
`export PYTHONPATH=[...]/facenet/src`
(Ensure to adjust to the path where you have downloaded FaceNet to.)
You can add this line to the bottom of your ~/.bashrc file to ensure that it is run every time you open a terminal.

Download a vesion of a pre trained FaceNet model. The one I used for my testing can be found [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk)

### Parameters

Run the project by executing `python3 project.py` with the following arguments:

- `--embedding`     Select a valid embedding implementation. (Valid options are "facenet" and "hog")
- `--classifier`    Select a valid classifier implementation. (Valid options are "svm" or "knn")
- `--dataset`       Provide full path to the dataset. (Only tested with lfw)
- `--min_nrof_images_per_class` Provide minimum number of images required for a class to be included
- `--num_test_images_per_class` Provide the number of test images required per class
- `--mdlpath`       Provide full path to the tensorflow facenet model (.pb file)
- `--gpu_memory_fraction`   Upper bound on the amount of GPU memory that will be used by the process

### Example

`python project.py --embedding facenet --classifier svm --dataset /<dir_to_dataset>/datasets/lfw/ --min_nrof_images_per_class 10 --num_test_images_per_class 3 --mdlpath /<dir_to_model>/models/20170514-110547/20170512-110547.pb`

## Results To Date

5 Nov 2017

|           | SVM     | Neural Net  |
|-----------|---------|-------------|
| Facenet    | 99.6%-30secs | - |
| HOG        | 30.4%-Overnight | - |
