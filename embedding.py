import cv2
import tensorflow as tf
import numpy as np
import facenet
import face_recognition
from skimage import exposure
from skimage import feature


class Embedding(object):
    
    def __init__(self, dataset):
        self.dataset = dataset

    def get_embeddings(self):
        pass

class FN_Embedding(Embedding):

    def __init__(self, dataset, mdlpath, imgsize, gpu_mem, batchsize=50):
        super().__init__(dataset)
        self.mdlpath = mdlpath
        self.imgsize = imgsize
        self.gpu_mem = gpu_mem
        self.batchsize = batchsize

    def get_embeddings(self):
        """
        get_embeddings - function to get embeddings of a dataset using FaceNet

        args    self

        returns data - array of embeddings for each image
                labels - array of corresponding labels for each embedding
        """
        results = dict()

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_mem)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                facenet.load_model(self.mdlpath)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                image_paths = []
                batch_num = 0

                data = []
                labels = []

                for cls in self.dataset:
                    for path in cls.image_paths:
                        image_paths.append(path)
                        labels.append(cls.name)

                for i in range(0, len(image_paths), self.batchsize):
                    print("batch num: %d of %d" % (i / self.batchsize, len(image_paths) / self.batchsize))

                    #load data
                    images = facenet.load_data(image_paths=image_paths[i:i+self.batchsize],
                            do_random_crop=False, do_random_flip=False, image_size=self.imgsize, do_prewhiten=True)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}

                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    for emb in emb_array:
                        data.append(emb)

                #convert list to numpy array for compatibility with the svm
                data = np.asarray(data)
                labels = np.asarray(labels)

                return data, labels

class HOG_OCV_Embedding(Embedding):

    def __init__(self, dataset):
        super().__init__(dataset)

    def get_embeddings(self):
        """
        get_embeddings - function to get embeddings (in form of HOG) of a dataset using OpenCV

        args    self

        returns data - array of embeddings for each image
                labels - array of corresponding labels for each embedding
        """

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
                img = cv2.resize(img, (250, 250))

                #convert image to greyscale
                #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                #calculate the hog for the image
                descriptor = hog.compute(img)

                #add data and label to respective lists
                data.append(descriptor)
                labels.append(cls.name)

        #convert list to numpy array for compatibility with the svm
        data = np.asarray(data)
        labels = np.asarray(labels)

        #return the array of hogs and labels
        return data, labels

class HOG_SKI_Embedding(Embedding):

    def __init__(self, dataset):
        super().__init__(dataset)

    def get_embeddings(self):
        """
        get_embeddings - function to get embeddings (in form of HOG) of a dataset using Scikit

        args    self

        returns data - array of embeddings for each image
                labels - array of corresponding labels for each embedding
        """

        data = []
        labels = []
        for cls in self.dataset:
            for image in cls.image_paths:
                img = cv2.imread(image)

                #convert image to greyscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                #calculate the hog for the image
                descriptor = feature.hog(gray_img, orientations=9, pixels_per_cell=(5,5),
                        cells_per_block=(2,2), transform_sqrt=True)

                #add data and label to respective lists
                data.append(descriptor)
                labels.append(cls.name)

        #convert list to numpy array for compatibility with the svm
        data = np.asarray(data)
        labels = np.asarray(labels)

        #return the array of hogs and labels
        return data, labels



class DLIBEmbedding(Embedding):
    """
    An implementation using Adam Geitgey's face_recognition library
    This is an API that makes use of DLIB

    args - dataset
    """
    def __init__(self, dataset):
        super().__init__(dataset)

    def get_embeddings(self):
        """
        get_embeddings - function to get embeddings using the face_recognition lib

        args    self

        returns data - array of embeddings for each image
                labels - array of corresponding labels for each embedding
        """

        data = []
        labels = []
        for cls in self.dataset:
            for image in cls.image_paths:
                #load image into an array
                img = face_recognition.load_image_file(image)

                #calculate face encodings for each face
                #(Grab index 0 because we know there is only one face per image)
                encoding = face_recognition.face_encodings(img)

                #add data and label to respective lists
                for i in encoding:
                    data.append(i)
                    labels.append(cls.name)

        #convert to numpy array for compatibility with the svm
        data = np.asarray(data)
        labels = np.asarray(labels)

        #return the arrays of encodings and labels
        return data, labels

