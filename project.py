import sys
import time
import argparse

from sklearn.externals import joblib

import facenet

import embedding
import classifier
import data_parse

def get_face_vectors(embed_type, dataset, modelpath, imgsize, gpu_mem):
    """
    get_face_vectors - function to provide facial embeddings for a dataset

    args    embed_type - type of embedding to find (hog or facenet)
            dataset - full path to dataset
            modelpath - full path to tensorflow facenet model (for embed_type facenet only)
            imgsize - size of image to use for facenet (for embed_type facenet only)

    returns data - array of facial vectors
            labels - array of labels that corresponds to the array of facial vectors
    """
    if embed_type == "hog_opencv":
        embed_method = embedding.HOG_OCV_Embedding(dataset)
    elif embed_type == "hog_scikit":
        embed_method = embedding.HOG_SKI_Embedding(dataset)
    elif embed_type == "facenet":
        embed_method = embedding.FN_Embedding(dataset, modelpath, imgsize, gpu_mem)
    elif embed_type == "dlib":
        embed_method = embedding.DLIBEmbedding(dataset)
    else:
        print("You have provided an invalid embedding type. (Valid options are facenet or hog)")
        return False

    data, labels = embed_method.get_embeddings()

    return data, labels

def classify(classify_type, trained_svm, train_data, train_labels, test_data, test_labels, num_classes):
    """
    classify - function to use facial embeddings to judge what label a face is associated with

    args    classify_type - type of classification to use ("svm" or "knn")
            train_data - data to use for training
            train_labels - labels to use for training
            test_data - data to use for testing
            test_labels - labels to check against predicted values

    returns accuracy - accuracy of the produced model
    """

    if classify_type == "svm":
        classify_method = classifier.SVM_Classifier(train_data, train_labels, test_data, test_labels)
    elif classify_type == "neural":
        classify_method = classifier.Neural_Classifier(train_data, train_labels, test_data, test_labels, num_classes)
    elif classify_type == "knn":
        classify_method = classifier.KNNClassifier(train_data, train_labels, test_data, test_labels)
    else:
        print("You have provided and invalid classifier type. (Valid options are svm or neural)")
        return False

    #if we are provided with a pre trained svm, there is no need to carry out training
    if trained_svm == "":
        model = classify_method.train()
    else:
        print("Using pre trained svm...")
        model = joblib.load(trained_svm)

    accuracy = classify_method.check_accuracy(model)

    return accuracy

def main(args):
    print("Parsing dataset...")
    dataset_tmp = facenet.get_dataset(args.dataset)
    train_set, test_set, num_classes = data_parse.split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.num_test_images_per_class)
    print("Dataset parsed.")

    start_time_embeddings = time.time()
    #Prepare Training Data
    print("Calculating training facial embeddings...")
    train_data, train_labels = get_face_vectors(args.embedding, train_set, args.mdlpath, args.imgsize, args.gpu_memory_fraction)
    int_train_labels, int_label_lookup_dict = data_parse.labels_to_int(train_labels)

    #Prepare Test Data
    print("Calculating testing facial embeddings...")
    test_data, test_labels = get_face_vectors(args.embedding, test_set, args.mdlpath, args.imgsize, args.gpu_memory_fraction)
    int_test_labels = data_parse.int_label_lookup(test_labels, int_label_lookup_dict)
    print("Embedding data fetched and now going to be classified.")
    print("Embedding time: %s minutes" % ((time.time() - start_time_embeddings)/60))

    #Run Classification
    if args.use_trained_svm == None:
        args.use_trained_svm = ""

    start_time_classify = time.time()
    result = classify(args.classifier, args.use_trained_svm, train_data, int_train_labels, test_data, int_test_labels, num_classes)

    print("Classify Time: %s minutes" % ((time.time() - start_time_classify)/60))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", help="Select method of getting facial embeddings (facenet, hog_opencv, hog_sklearn or dlib)", type=str, required=True)
    parser.add_argument("--classifier", help="Select method of classifying images (svm or knn)", type=str, required=True)
    parser.add_argument("--dataset", help="Full path to dataset dir", type=str, required=True)
    parser.add_argument("--min_nrof_images_per_class", help="minimum images needed for a class to be included", type=int, required=True)
    parser.add_argument("--num_test_images_per_class", help="number of test images per class", type=int, required=True)
    parser.add_argument("--mdlpath", help="Full path to tensorflow model to use", type=str, required=False)
    parser.add_argument("--imgsize", help="Size of images to use", type=int, default=160, required=False)
    parser.add_argument("--gpu_memory_fraction", help="tensorflow gpu memory usage", type=float, required=False)
    parser.add_argument("--use_trained_svm", help="path to pre trained svm", type=str, required=False)
    args = parser.parse_args()   
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
