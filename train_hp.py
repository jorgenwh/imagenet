import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from alexnet import AlexNet
from resnet import resnet
from data_loader import get_data_loader
from helpers import AverageMeter

LOAD_MODEL = None
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
NUM_CLASSES = 1000
BATCH_SIZE = 128
EPOCHS = 90
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = AlexNet(
#         input_channels=IMAGE_CHANNELS,
#         input_height=IMAGE_SIZE,
#         input_width=IMAGE_SIZE,
#         num_classes=NUM_CLASSES
# )
model = resnet(
        num_layers=18,
        input_channels=IMAGE_CHANNELS,
        input_height=IMAGE_SIZE,
        input_width=IMAGE_SIZE,
        num_classes=NUM_CLASSES
)

if LOAD_MODEL is not None:
    model.load_state_dict(torch.load(LOAD_MODEL))
model = model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
loss_fn = torch.nn.CrossEntropyLoss()

train_loader, val_loader = get_data_loader(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

# Create a GradScaler for mixed precision training
scaler = GradScaler()

for epoch in range(EPOCHS):
    print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS))

    train_loss = AverageMeter()
    train_accuracy = 0
    train_images_seen = 0

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # Use autocast to enable mixed-precision training
        with autocast():
            # forward pass
            output = model(images)

            # compute loss
            loss = loss_fn(output, labels)
            train_loss.update(loss.item(), images.size(0))

            # backward pass
            scaler.scale(loss).backward()

            # compute accuracy
            output = torch.argmax(output, dim=1)
            correctly_predicted = torch.sum(output == labels).item()
            train_accuracy += correctly_predicted
            train_images_seen += images.size(0)

            scaler.step(optimizer)
            scaler.update()

            print(
                "batch: " + str(i + 1) + "/" + str(len(train_loader)) + " - " +
                "train_loss: " + str(train_loss) + " - " +
                "train_accuracy: " + str(round(train_accuracy / train_images_seen, 4)) + " " * 10, end="\r"
            )

    # validation
    val_loss = AverageMeter()
    val_accuracy = 0
    val_images_seen = 0

    model.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward pass
        with torch.no_grad(), autocast():
            output = model(images)

        # compute loss
        loss = loss_fn(output, labels)
        val_loss.update(loss.item(), images.size(0))

        # compute accuracy
        output = torch.argmax(output, dim=1)
        correctly_predicted = torch.sum(output == labels).item()
        val_accuracy += correctly_predicted
        val_images_seen += images.size(0)

    # adjust learning rate
    lr = LEARNING_RATE * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print(
        "batch: " + str(i + 1) + "/" + str(len(train_loader)) + " - " +
        "train_loss: " + str(train_loss) + " - " +
        "train_accuracy: " + str(round(train_accuracy / train_images_seen, 4)) + " - " +
        "val_loss: " + str(val_loss) + " - " +
        "val_accuracy: " + str(round(val_accuracy / val_images_seen, 4)) + " " * 10
    )

