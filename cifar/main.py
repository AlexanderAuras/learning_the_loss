import torch
from torch import nn
import torchvision
from torchvision import transforms

from resnet18 import ResNet18
from simple_net import SimpleCIFARNet

def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to("cuda"), y.to("cuda")
        z = model(x)
        loss = loss_function(z, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validate(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to("cuda"), y.to("cuda")
            z = model(x)
            loss += loss_function(z, y).item()
            correct += (z.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"\nValidation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")

def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to("cuda"), y.to("cuda")
            z = model(x)
            loss += loss_function(z, y).item()
            correct += (z.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")

#transforms.Resize((224, 224)), 
data = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
training_data, validation_data = torch.utils.data.random_split(data, [int(len(data)*0.75), int(len(data)*0.25)])
test_data = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64)
validate_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)

model = SimpleCIFARNet()#ResNet18()
#model.load_state_dict(torch.load("resnet18.pth"))
model.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(7):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_function, optimizer)
    validate(validate_dataloader, model, loss_function)
test(test_dataloader, model, loss_function)

torch.save(model.state_dict(), "simple_net.pth")