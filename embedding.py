import tensorflow as tf
import numpy as np

import facenet


class Embedding(object):
    
    def __init__(self, dataset):
        self.dataset = dataset

    def get_embeddings(self):
        pass

class FN_Embedding(Embedding):

    def __init__(self, dataset, mdlpath, imgsize, batchsize=200):
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
        #values and setup for the HOG Descriptor
        win_size = (250, 250)
        block_size = (10, 10)
        cell_size = (5, 5)
        block_stride = (5, 5)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4.
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64

        hog = cv2.HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins,deriv_aperture,win_sigma,
                                        histogram_norm_type,l2_hys_threshold,gamma_correction,nlevels)

        data = []
        labels = []
        for cls in self.dataset:
            for image in cls.image_paths:
                img = cv2.imread(image)

                #convert image to greyscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                #calculate the hog for the image
                descriptor = hog.compute(gray_img)

                #add data and label to respective arrays
                data.append(descriptor)
                labels.append(cls.name)

                flipped_img = cv2.flip(gray_img, 0)

                descriptor_2 = hog.compute(flipped_img)

                #add data and label to respective arrays
                data.append(descriptor_2)
                labels.append(cls.name)

        #convert list to numpy array for compatibility with the svm
        data = np.asarray(data)
        labels = np.asarray(labels)

        #return the array of hogs and labels
        return data, labels




