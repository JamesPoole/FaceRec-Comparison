import argparse

import embedding

def get_face_vectors(embed_type):
    if embed_type is "hog":
        embed_method = embedding.HOG_Embedding()
    elif embed_type is "facenet":
        embed_method = embedding.FN_Embedding()
    else:
        print("You have provided an invalid embedding type. (Valid options are facenet or hog)")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", help="Select method of getting facial embeddings (facenet or hog)", type=str, required=True)

    get_face_vectors(args.embedding)
