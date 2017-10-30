import tensorflow as tf
import numpy as np

import facenet


class Embedding(object):
    
    def __init__(self, dataset):
        self.dataset = dataset

    def get_embeddings(self):
        pass

class FN_Embedding(Embedding):

    def __init__(self, dataset, mdlpath, imgsize, batchsize):
        super().__init__(dataset)
        self.mdlpath = mdlpath
        self.imgsize = imgsize
        self.batchsize = batchsize

    def get_embeddings(self):
        with tf.Graph.as_default():
            with tf.Session() as sess:
                facenet.load_model(mdlpath)


class HOG_Embedding(Embedding):

    def __init__(self, dataset):
        super().__init__(dataset)

    def get_embeddings(self):
        pass
