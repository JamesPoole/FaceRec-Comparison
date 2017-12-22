import facenet
import numpy as np

def split_dataset(dataset, min_nrof_images_per_class, nrof_test_images_per_class):
    """
    split_dataset - function to split the dataset into a train set and a test set

    args    dataset - dataset to be split
            min_nrof_images_per_class - minimum num of images required for a class to be used
            nrof_train_images_per_class - num of images used for training within a class

    returns train_set - dataset for training
            test_set - dataset for testing
    """

    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            test_set.append(facenet.ImageClass(cls.name, paths[:nrof_test_images_per_class]))
            train_set.append(facenet.ImageClass(cls.name, paths[nrof_test_images_per_class:]))

    print('Classes: %d' % (len(test_set)))

    return train_set, test_set

def labels_to_int(labels):
    """
    labels_to_int - function to convert labels to ints because svm will not accept strings as labels

    args    labels - array of labels to be converted

    returns int_labels - array of labels in int form
            int_label_dict - dictionary for easy lookup of name to int
    """

    current_name = ""
    current_int = 0
    int_labels = []
    int_label_dict ={}
    for name in labels:
        if name != current_name:
            current_name = name
            int_labels.append(current_int)
            int_label_dict.update({current_name: current_int})
            current_int += 1
        else:
            int_labels.append(current_int)

    int_labels[0] = 1
    int_labels = np.asarray(int_labels)

    #returns the label array in ints and a dictionary to easily look up what number belongs to what person
    return int_labels, int_label_dict

def int_label_lookup(test_labels, int_label_dict):
    """
    int_label_lookup -  function to convert test labels to ints.
                        when given a test set, the data is randomised for fairness
                        but we need to make sure to match the correct labels to the correct ints.

    args    test_labels - labels for use in test
            int_label_dict - dictionary for easy label int lookup

    return  int_labels - test labels converted to ints
    """

    int_labels = []
    for name in test_labels:
        int_labels.append(int_label_dict[name])  

    return int_labels
