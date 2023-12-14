import numpy as np
import cv2
import pickle
import random

with open("data/preprocessed/indices.pickle", "rb") as f:
    imagenet_indices = pickle.load(f)

DATASET = "val"
CHUNK = 1
IMG_SIZE = 224

images = np.load("data/preprocessed/" + DATASET + "/" + DATASET + "_image_chunk_" + str(CHUNK) + ".npy")
labels = np.load("data/preprocessed/" + DATASET + "/" + DATASET + "_label_chunk_" + str(CHUNK) + ".npy")

# display random images along with their labels, both in integer and string form
while True:
    rand_idx = random.randint(0, len(images) - 1)
    image = images[rand_idx]
    image = np.transpose(image, (1, 2, 0))
    label = labels[rand_idx]

    string_label = None
    for key, value in imagenet_indices.items():
        if value == label:
            string_label = key
            break

    print(str(label) + " - " + string_label)

    cv2.imshow("Image", image)
    k = cv2.waitKey()

    # exit if q or esc is pressed
    if k == ord("q") or k == 27:
        break

cv2.destroyAllWindows()
