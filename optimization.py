import torch, time
from torch import nn
from format_image import FormatImage
from dataset import FoodDataset
from torch.utils.data import DataLoader
from model import NeuralNetwork


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    t0 = time.time()
    first = True

    for batch, (X, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X, y = X.to("cuda"), y.to("cuda")

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print time calcuations for only first batch
        if first:
            projected = int((time.time() - t0) * len(dataloader))
            hours = projected // 3600
            minutes = (projected - (hours * 3600)) // 60
            print(f"Batch time: {time.time()-t0:>0.1f}s,", end=" ")
            print(f"Projected epoch time: {hours} hours, {minutes} minutes")
            first = False

        # Print info every couple batches
        if (batch + 1) % 8 == 0:
            loss = loss.item()
            if (batch + 1) < len(dataloader):
                current = (batch + 1) * BATCH_SIZE
            else:
                current = size
            print(f"Loss: {loss:>7f}, [{current:>5d}/{size:>5d}],", end=" ")
            print(f"Elapsed time: {time.time()-t0:>0.1f}s")


def test_loop(dataloader, model, loss_fn):
    # Disable dropout layers and stuff, N/A for this model
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    t0 = time.time()

    with torch.no_grad():
        for X, y in dataloader:
            if torch.cuda.is_available():
                X, y = X.to("cuda"), y.to("cuda")

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # Check one hot encoding with max argument
            correct += (pred.argmax(1) == y).int().sum().item()

    test_loss /= num_batches
    correct /= size

    print("Test Error:")
    print(f"Accuracy: {(100*correct):>0.1f}%,", end=" ")
    print(f"Avg loss: {test_loss:>8f},", end=" ")
    print(f"Test time: {time.time()-t0:>0.1f}s\n")


IMAGES_DIR = "data\\images\\"
TRAIN_DATA_PATH = "data\\train.csv"
TEST_DATA_PATH = "data\\test.csv"

IMAGE_SIZE = 256

LEARNING_RATE = 1e-5
BATCH_SIZE = 128
EPOCHS = 1000


start = 7
start_path = f"models\\all50-{IMAGE_SIZE}"

model = NeuralNetwork()
model.load_state_dict(torch.load(f"{start_path}\\all{start}(79.7%).pth"))

if torch.cuda.is_available():
    model = model.to("cuda")

for param in model.parameters():
    param.requires_grad_(True)

training_data = FoodDataset(TRAIN_DATA_PATH, IMAGES_DIR, FormatImage(IMAGE_SIZE))
test_data = FoodDataset(TEST_DATA_PATH, IMAGES_DIR, FormatImage(IMAGE_SIZE))

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer.load_state_dict(torch.load(f"{start_path}\\optimizer{start}.pth"))

for t in range(start, EPOCHS):
    print(f"Epoch {t+1}\n---------------------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    torch.save(model.state_dict(), f"{start_path}\\all{t+1}.pth")
    torch.save(optimizer.state_dict(), f"{start_path}\\optimizer{t+1}.pth")
    test_loop(test_dataloader, model, loss_fn)

print("Done!")
