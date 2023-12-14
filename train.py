import torch
import torch.nn.functional as F
import numpy as np

from cifar.models import ImagenetConvNet, AlexNet

IMAGENET_BASE_PATH = "data/preprocessed/"
TRAIN_PATH = IMAGENET_BASE_PATH + "train/"
VAL_PATH = IMAGENET_BASE_PATH + "val/"
NUM_TRAIN_CHUNKS = 8
NUM_VAL_CHUNKS = 1

LOAD_MODEL = None # set to path of model to load
BATCH_SIZE = 96
EPOCHS = 60
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model and load parameter checkpoint if LOAD_MODEL is defined
model = AlexNet()
if LOAD_MODEL is not None:
    model.load_state_dict(torch.load(LOAD_MODEL))
model = model.to(DEVICE)

# create optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# training loop
for epoch in range(EPOCHS):
    print("Epoch: " + str(epoch+1) + "/" + str(EPOCHS))

    train_loss = 0
    train_accuracy = 0
    images_processed = 0
    model.train()

    X = None
    y = None

    for chunk in range(NUM_TRAIN_CHUNKS):
        X = np.load(TRAIN_PATH + "train_image_chunk_" + str(chunk + 1) + ".npy")
        y = np.load(TRAIN_PATH + "train_label_chunk_" + str(chunk + 1) + ".npy")
        n_train_batches = len(X)//BATCH_SIZE + 1

        for i in range(0, len(X), BATCH_SIZE):
            x_batch = torch.tensor(X[i:i+BATCH_SIZE].astype(np.float32)/255.0).to(DEVICE)
            y_batch = torch.tensor(y[i:i+BATCH_SIZE]).to(DEVICE)

            # forward pass
            y_pred = model(x_batch)

            # compute loss
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute accuracy
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            train_accuracy += torch.sum(y_pred == y_batch).item()

            images_processed += len(x_batch)

            print(
                "chunk: " + str(chunk + 1) + "/" + str(NUM_TRAIN_CHUNKS) + " - " +
                "batch: " + str(i//BATCH_SIZE + 1) + "/" + str(n_train_batches) + " - " +
                "train_loss: " + str(round(train_loss/images_processed, 4)) + " - " +
                "train_accuracy: " + str(round(train_accuracy/images_processed, 4)) + " "*10, end="\r"
            )

        del X
        del y


    # compute validation loss and accuracy
    val_loss = 0
    val_accuracy = 0
    model.eval()

    X = np.load(VAL_PATH + "val_image_chunk_1.npy")
    y = np.load(VAL_PATH + "val_label_chunk_1.npy")
    n_val_batches = len(X)//BATCH_SIZE + 1

    for i in range(0, len(X), BATCH_SIZE):
        x_batch = torch.tensor(X[i:i+BATCH_SIZE].astype(np.float32)/255.0).to(DEVICE)
        y_batch = torch.tensor(y[i:i+BATCH_SIZE]).to(DEVICE)

        # forward pass
        y_pred = model(x_batch)

        # compute loss
        loss = loss_fn(y_pred, y_batch)
        val_loss += loss.item()

        # compute accuracy
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        val_accuracy += torch.sum(y_pred == y_batch).item()

    print(
        "chunk: " + str(chunk + 1) + "/" + str(NUM_TRAIN_CHUNKS) + " - " +
        "train_batch: " + str(i//BATCH_SIZE + 1) + "/" + str(n_train_batches) + " - " +
        "train_loss: " + str(round(train_loss/images_processed, 4)) + " - " +
        "train_accuracy: " + str(round(train_accuracy/images_processed, 4)) + " - " +
        "val_loss: " + str(round(val_loss/len(X), 4)) + " - " +
        "val_accuracy: " + str(round(val_accuracy/len(X), 4))
    )

    del X
    del y
