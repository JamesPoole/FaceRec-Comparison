#My Final Year Project - A Comparison of Facial Recognition Implementations

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

##Implemented Features
1) Dataset Parsing
2) Obtaining HOG Feature Vectors
3) SVM

##Features still to be Implemented
1) Obtaining FaceNet Embeddings
2) Neural Classifier
3) Dataset Facial Alignment
4) Improved Performance Measuring

##Prerequisites
This requires Python 3.X

Python requirements can be found and installed from requirements.txt
Tested using the ([LFW Dataset](http://vis-www.cs.umass.edu/lfw/)). You can download this ([here](http://vis-www.cs.umass.edu/lfw/lfw.tgz)).

###Parameters
- `--embedding`     Select a valid embedding implementation. (Currently only working argument is "hog")
- `--classifier`    Select a valid classifier implementation. (Currently only working argument is "svm")
- `--dataset`       Provide full path to the dataset. (Only tested with lfw)


