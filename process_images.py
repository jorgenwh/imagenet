import os
import time
import numpy as np
import cv2
import pickle

from classes import IMAGENET2012_CLASSES

DATASET = "train"

imagenet_indices = {IMAGENET2012_CLASSES[k]:i for i,k in enumerate(IMAGENET2012_CLASSES.keys())}
filenames = os.listdir(f"data/raw/{DATASET}")

NUM_IMAGES = len(filenames)
IMG_SIZE = 224
CHUNK_SIZE = int(2e9/(3*IMG_SIZE*IMG_SIZE)) # 5 GB
CHUNK_BYTE_SIZE = CHUNK_SIZE*3*IMG_SIZE*IMG_SIZE 

remaining_images = NUM_IMAGES
size = min(CHUNK_SIZE, NUM_IMAGES)*3*IMG_SIZE*IMG_SIZE
IMAGE_CHUNK = np.zeros(
        size, dtype=np.uint8
).reshape(
        min(CHUNK_SIZE, NUM_IMAGES), 3, IMG_SIZE, IMG_SIZE
)
LABEL_CHUNK = np.zeros(min(CHUNK_SIZE, NUM_IMAGES), dtype=np.int64)

processed_chunks = 0
for i, fn in enumerate(filenames):
    print(
        "Processing chunk: " + str(processed_chunks + 1) + "/" + str(NUM_IMAGES//CHUNK_SIZE + 1) + " - " +
        "Image: " + str((i + 1)%CHUNK_SIZE) + "/" + str(IMAGE_CHUNK.shape[0]) + " - " +
        "Remaining images: " + str(remaining_images) + " "*10,
        end="\r"
    )

    img = cv2.imread("data/raw/" + DATASET + "/" + fn)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.transpose(img, (2, 0, 1))
    IMAGE_CHUNK[i%CHUNK_SIZE] = img

    if DATASET == "train":
        label_id = fn.split("_")[0]
    if DATASET == "val":
        label_id = fn.split(".")[0].split("_")[3]
    if DATASET == "test":
        raise NotImplementedError("Parsing test filenames not implemented")

    label = IMAGENET2012_CLASSES[label_id]
    label_index = imagenet_indices[label]
    LABEL_CHUNK[i%CHUNK_SIZE] = label_index

    if i % CHUNK_SIZE == 0 and i != 0:
        np.save("data/preprocessed/" + DATASET + "/" + DATASET + "_image_chunk_" + str(processed_chunks + 1) + ".npy", IMAGE_CHUNK)
        np.save("data/preprocessed/" + DATASET + "/" + DATASET + "_label_chunk_" + str(processed_chunks + 1) + ".npy", LABEL_CHUNK)
        size = min(CHUNK_SIZE, remaining_images)*3*IMG_SIZE*IMG_SIZE
        IMAGE_CHUNK = np.zeros(
                size, dtype=np.uint8
        ).reshape(
                min(CHUNK_SIZE, remaining_images), 3, IMG_SIZE, IMG_SIZE
        )
        LABEL_CHUNK = np.zeros(min(CHUNK_SIZE, remaining_images), dtype=np.int64)
        processed_chunks += 1

        print(
            "Processing chunk: " + str(processed_chunks) + "/" + str(NUM_IMAGES//CHUNK_SIZE + 1) + " - " +
            "Image: " + str(CHUNK_SIZE) + "/" + str(IMAGE_CHUNK.shape[0]) + " - " +
            "Remaining images: " + str(remaining_images) + " "*10,
        )
    elif i == NUM_IMAGES - 1:
        np.save("data/preprocessed/" + DATASET + "/" + DATASET + "_image_chunk_" + str(processed_chunks + 1) + ".npy", IMAGE_CHUNK)
        np.save("data/preprocessed/" + DATASET + "/" + DATASET + "_label_chunk_" + str(processed_chunks + 1) + ".npy", LABEL_CHUNK)
        processed_chunks += 1

        print(
            "Processing chunk: " + str(processed_chunks) + "/" + str(NUM_IMAGES//CHUNK_SIZE + 1) + " - " +
            "Image: " + str(i + 1) + "/" + str(IMAGE_CHUNK.shape[0]) + " - " +
            "Remaining images: " + str(remaining_images) + " "*10,
        )

    remaining_images -= 1
