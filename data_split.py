import facenet
import numpy as np

"""
split_dataset - function to split the dataset into a train set and a test set

args    dataset - dataset to be split
        min_nrof_images_per_class - minimum num of images required for a class to be used
        nrof_train_images_per_class - num of images used for training within a class

returns train_set - dataset for training
        test_set - dataset for testing
"""
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


"""
get_train_test_set - function to retrieve data set from location and call the split function

args    data_dir - full path to dataset directory

returns train_set - dataset for training
        test_set - dataset for testing
"""
def get_train_test_set(data_dir):
    dataset = facenet.get_dataset(data_dir)

    paths, labels = facenet.get_image_paths_and_labels(dataset)
    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))

    #split the dataset into train and test set
    train_set, test_set = split_dataset(dataset, 5, 4)
    return train_set, test_set
