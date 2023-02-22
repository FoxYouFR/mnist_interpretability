import yaml
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

config = yaml.safe_load(open("config.yaml"))

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=config["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(f"Number of available GPUs: {torch.cuda.device_count()}")
print(f"NCCL is available: {torch.distributed.is_nccl_available()}")
print(f"MPI is available: {torch.distributed.is_mpi_available()}")
print(f"GLOO is available: {torch.distributed.is_gloo_available()}")

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, config["conv1_out"], kernel_size=config["kernel_size"])
        self.conv2 = nn.Conv2d(config["conv1_out"], config["conv2_out"], kernel_size=config["kernel_size"])
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, config["fc1_out"])
        self.fc2 = nn.Linear(config["fc1_out"], 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=config["dropout_rate"], training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if batch % config["log_interval"] == 0:
            print("Train: {}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                batch * len(X), len(dataloader.dataset), 100. * batch / len(dataloader), loss.item()
            ))
            torch.save(model.state_dict(), config["model_save_file"])

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            test_loss += loss_fn(output, y).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()
    size = len(dataloader.dataset)
    test_loss /= size
    print(f"Test Error: \n Accuracy: {(100. * correct / size):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def load_and_test(model, loss_fn, optimizer, using_saved=False):
    if not using_saved:
        for t in range(config["epochs"]):
            print(f"Epoch: {t+1}\n----------------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test(test_dataloader, model, loss_fn)
        print("DONE")
    else:
        model = Network()
        model.load_state_dict(torch.load(config["model_save_file"]))
        model = model.to(device)

def feature_viz(model, dataloader):
    model.to("cpu")
    children = list(model.children())
    nb_layers = 0
    conv_layers = []
    for child in children:
        if type(child) == nn.Conv2d:
            nb_layers += 1
            conv_layers.append(child)

    print(nb_layers)
    examples = enumerate(dataloader)
    _, (data, targets) = next(examples)
    img = data[0][0]
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title(f"Ground Truth: {targets[0]}")
    plt.show()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))
    outputs = results
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(20, 20))
        layer_viz = outputs[num_layer][0, :, :, :]
        print(f"Layer {num_layer + 1} with shape {layer_viz.shape}")
        layer_viz = layer_viz.data
        for i, filter in enumerate(layer_viz):
            if i == 20: break
            plt.subplot(4, 5, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        plt.show()
        plt.close()

model = Network().to(device)
print(model)

loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
load_and_test(model, loss_fn, optimizer, using_saved=True)
feature_viz(model, test_dataloader)