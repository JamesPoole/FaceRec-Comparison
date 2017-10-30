import sys
import argparse

import embedding

def get_face_vectors(embed_type, dataset, modelpath, imgsize):
    if embed_type == "hog":
        embed_method = embedding.HOG_Embedding(dataset)
    elif embed_type == "facenet":
        embed_method = embedding.FN_Embedding(dataset, modelpath, imgsize)
    else:
        print("You have provided an invalid embedding type. (Valid options are facenet or hog)")
        return False

def main(args):
    get_face_vectors(args.embedding, args.dataset, args.mdlpath, args.imgsize)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", help="Select method of getting facial embeddings (facenet or hog)", type=str, required=True)
    parser.add_argument("--classifier", help="Select method of classifying images (neural or svm)", type=str, required=True)
    parser.add_argument("--dataset", help="Full path to dataset dir", type=str, required=True)
    parser.add_argument("--mdlpath", help="Full path to tensorflow model to use", type=str, required=False)
    parser.add_argument("--imgsize", help="Size of images to use", type=int, default=160, required=False)

    args = parser.parse_args()   
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
