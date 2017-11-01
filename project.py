import sys
import argparse

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
    if embed_type == "hog":
        embed_method = embedding.HOG_Embedding(dataset)
    elif embed_type == "facenet":
        embed_method = embedding.FN_Embedding(dataset, modelpath, imgsize, gpu_mem)
    else:
        print("You have provided an invalid embedding type. (Valid options are facenet or hog)")
        return False

    data, labels = embed_method.get_embeddings()

    return data, labels

def classify(classify_type, train_data, train_labels, test_data, test_labels):
    if classify_type == "svm":
        classify_method = classifier.SVM_Classifier(train_data, train_labels, test_data, test_labels)
    elif classify_type == "neural":
        classify_method = classifier.Neural_Classifier()
    else:
        print("You have provided and invalid classifier type. (Valid options are svm or neural)")
        return False

    model = classify_method.train()
    response = classify_method.test(model)
    accuracy = classify_method.check_accuracy(response)

    return accuracy

def main(args):
    train_set, test_set = data_parse.get_train_test_set(args.dataset)
    print("Parsing dataset...")

    #Prepare Training Data
    train_data, train_labels = get_face_vectors(args.embedding, train_set, args.mdlpath, args.imgsize, args.gpu_memory_fraction)
    int_train_labels, int_label_lookup_dict = data_parse.labels_to_int(train_labels)
    print("Training data parsed.")

    #Prepare Test Data
    test_data, test_labels = get_face_vectors(args.embedding, test_set, args.mdlpath, args.imgsize)
    int_test_labels = data_parse.int_label_lookup(test_labels, int_label_lookup_dict)
    print("Test data parsed.")

    result = classify(args.classifier, train_data, int_train_labels, test_data, int_test_labels)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", help="Select method of getting facial embeddings (facenet or hog)", type=str, required=True)
    parser.add_argument("--classifier", help="Select method of classifying images (neural or svm)", type=str, required=True)
    parser.add_argument("--dataset", help="Full path to dataset dir", type=str, required=True)
    parser.add_argument("--mdlpath", help="Full path to tensorflow model to use", type=str, required=False)
    parser.add_argument("--imgsize", help="Size of images to use", type=int, default=160, required=False)
    parser.add_argument("--gpu_memory_fraction", help="tensorflow gpu memory usage", type=float, required=False)

    args = parser.parse_args()   
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
