seed = 0

import io
import itertools
import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import torch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms

import higher

import matplotlib.pyplot as plt
from PIL import Image

from resnet18 import ResNet18
from simple_net import SimpleCIFARNet
from label_smoothing_loss import LabelSmoothingLoss



def plot_matrix(matrix):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.inferno)
    plt.title("Smoothing matrix")
    plt.colorbar()
    normalized = (matrix-matrix.min())/(matrix.max()-matrix.min())
    matrix = np.around(matrix, decimals=2)
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i,j], horizontalalignment="center", color=("white" if normalized[i,j] < 0.5 else "black"))
    plt.tight_layout()
    plt.xticks(np.arange(matrix.shape[0]), np.arange(matrix.shape[0]))
    plt.yticks(np.arange(matrix.shape[1]), np.arange(matrix.shape[1]))
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(figure)
    buffer.seek(0)
    return torchvision.transforms.ToTensor()(Image.open(buffer))



def train(model: nn.Module, 
            training_dataloader: torch.utils.data.DataLoader,
            inner_loss_function: nn.Module, 
            inner_optimizer: torch.optim.Optimizer, 
            validation_dataloader: torch.utils.data.DataLoader, 
            outer_loss_function: nn.Module, 
            outer_optimizer: torch.optim.Optimizer)->None:
    global global_epoch, global_steps_per_epoch, global_step_in_epoch
    lastPercent = 0.0
    model.train()
    print(f"{global_epoch}: 0%", end=" ", flush=True)
    for i in range(global_steps_per_epoch):
        if (i/float(max(global_steps_per_epoch-1, 1)))*100.0 >= lastPercent+10.0:
            lastPercent += 10.0
            print(f"{int(lastPercent)}%", end=" ", flush=True)
        global_step_in_epoch = i
        #step(model, training_dataloader, inner_loss_function, inner_optimizer)
        bilevel_step(model, training_dataloader, inner_loss_function, inner_optimizer, validation_dataloader, outer_loss_function, outer_optimizer)
    print()

    

def step(model: nn.Module,
            training_dataloader: torch.utils.data.DataLoader,
            loss_function: nn.Module, 
            optimizer: torch.optim.Optimizer)->None:
    global global_epoch, global_steps_per_epoch, global_step_in_epoch

    _, (train_x, train_y) = next(enumerate(training_dataloader))
    train_x, train_y = train_x.to("cuda"), train_y.to("cuda")
    train_z = model(train_x)
    loss = loss_function(train_z, train_y)
    probabilities = train_z.softmax(dim=1)
    global_logger.add_scalar("train/inner-loss", loss.item(), global_epoch*global_steps_per_epoch+global_step_in_epoch)
    right_confidence = probabilities[probabilities.argmax(1)==train_y].max(dim=1)[0].mean().item()
    wrong_confidence = probabilities[probabilities.argmax(1)!=train_y].max(dim=1)[0].mean().item()
    global_logger.add_scalar("train/inner-right-confidence", right_confidence, global_epoch*global_steps_per_epoch+global_step_in_epoch)
    global_logger.add_scalar("train/inner-wrong-confidence", wrong_confidence, global_epoch*global_steps_per_epoch+global_step_in_epoch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def bilevel_step(model: nn.Module, 
                    training_dataloader: torch.utils.data.DataLoader,
                    inner_loss_function: nn.Module, 
                    inner_optimizer: torch.optim.Optimizer, 
                    validation_dataloader: torch.utils.data.DataLoader, 
                    outer_loss_function: nn.Module, 
                    outer_optimizer: torch.optim.Optimizer)->None:
    global global_epoch, global_steps_per_epoch, global_step_in_epoch, global_inner_iterations

    with higher.innerloop_ctx(model, inner_optimizer) as (fmodel, diffopt):
        ######Inner optimization#####
        inner_loss_acc = 0
        inner_right_confidence = 0
        inner_right_confidence_count = 0
        inner_wrong_confidence = 0
        inner_wrong_confidence_count = 0
        for i, (train_x, train_y) in enumerate(training_dataloader):
            if i >= global_inner_iterations:
                break
            train_x, train_y = train_x.to("cuda"), train_y.to("cuda")
            train_z = fmodel(train_x)
            inner_loss = inner_loss_function(train_z, train_y)
            inner_loss_acc += inner_loss.item()
            with torch.no_grad():
                probabilities = train_z.softmax(dim=1)
                if probabilities[probabilities.argmax(1)==train_y].shape[0] > 0:
                    inner_right_confidence += probabilities[probabilities.argmax(1)==train_y].max(dim=1)[0].mean()
                    inner_right_confidence_count += 1
                if probabilities[probabilities.argmax(1)!=train_y].shape[0] > 0:
                    inner_wrong_confidence += probabilities[probabilities.argmax(1)!=train_y].max(dim=1)[0].mean()
                    inner_wrong_confidence_count += 1
            diffopt.step(inner_loss)
        global_logger.add_scalar("train/inner-loss", inner_loss_acc/global_inner_iterations, global_epoch*global_steps_per_epoch+global_step_in_epoch)
        global_logger.add_scalar("train/inner-right-confidence", inner_right_confidence/inner_right_confidence_count if inner_right_confidence_count else float("NaN"), global_epoch*global_steps_per_epoch+global_step_in_epoch)
        global_logger.add_scalar("train/inner-wrong-confidence", inner_wrong_confidence/inner_wrong_confidence_count if inner_wrong_confidence_count else float("NaN"), global_epoch*global_steps_per_epoch+global_step_in_epoch)
        new_model_state = fmodel.state_dict()
        new_optimizer_state = diffopt.state[0]
        #############################

        #######Outer optimization#####
        _, (val_x, val_y) = next(enumerate(validation_dataloader))
        val_x, val_y = val_x.to("cuda"), val_y.to("cuda")
        val_z = fmodel(val_x)
        outer_loss = outer_loss_function(val_z, val_y)
        global_logger.add_scalar("train/outer-loss", outer_loss.item(), global_epoch*global_steps_per_epoch+global_step_in_epoch)
        outer_probabilities = val_z.softmax(dim=1)
        global_logger.add_scalar("train/outer-right-confidence", outer_probabilities[outer_probabilities.argmax(1)==val_y].max(dim=1)[0].mean().item(), global_epoch*global_steps_per_epoch+global_step_in_epoch)
        global_logger.add_scalar("train/outer-wrong-confidence", outer_probabilities[outer_probabilities.argmax(1)!=val_y].max(dim=1)[0].mean().item(), global_epoch*global_steps_per_epoch+global_step_in_epoch)
        outer_optimizer.zero_grad()
        outer_loss.backward()
        outer_optimizer.step()
        ##############################

        #global_logger.add_scalar("train/smoothing-parameter", inner_loss_function.smoothing.item(), global_epoch*global_steps_per_epoch+global_step_in_epoch)
        #global_logger.add_image("train/smoothing-parameter", plot_matrix(inner_loss_function.smoothing.clone().detach().cpu().numpy()), global_epoch*global_steps_per_epoch+global_step_in_epoch)
        
    ##########Copy data###########
    with torch.no_grad():
        model.load_state_dict(new_model_state)
    with torch.no_grad():
        for group_idx, entries in new_optimizer_state.items():
            for entry_key, entry_value in entries.items():
                if torch.is_tensor(entry_value):
                    inner_optimizer.state[inner_optimizer.param_groups[0]["params"][group_idx]][entry_key].copy_(entry_value)
                else:
                    inner_optimizer.state[inner_optimizer.param_groups[0]["params"][group_idx]][entry_key] = entry_value
    ##############################



def validate(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_function: nn.Module)->None:
    global global_epoch, global_steps_per_epoch
    model.eval()
    val_loss_acc = 0.0
    confusion_matrix = torch.zeros((10,10), dtype=torch.int32)
    right_confidence = 0
    right_confidence_count = 0
    wrong_confidence = 0
    wrong_confidence_count = 0
    with torch.no_grad():
        for _, (val_x, val_y) in enumerate(dataloader):
            val_x, val_y = val_x.to("cuda"), val_y.to("cuda")
            val_z = model(val_x)
            confusion_matrix.put_(val_y.cpu().long()*10+val_z.cpu().argmax(1).long(), torch.ones((val_x.shape[0]), dtype=torch.int), accumulate=True)
            val_loss_acc += loss_function(val_z, val_y).item()
            with torch.no_grad():
                probabilities = val_z.softmax(dim=1)
                if probabilities[probabilities.argmax(1)==val_y].shape[0] > 0:
                    right_confidence += probabilities[probabilities.argmax(1)==val_y].max(dim=1)[0].mean()
                    right_confidence_count += 1
                if probabilities[probabilities.argmax(1)!=val_y].shape[0] > 0:
                    wrong_confidence += probabilities[probabilities.argmax(1)!=val_y].max(dim=1)[0].mean()
                    wrong_confidence_count += 1
    global_logger.add_scalar("validation/loss", val_loss_acc/len(dataloader), (global_epoch+1)*global_steps_per_epoch)
    global_logger.add_scalar("validation/accuracy", confusion_matrix.diagonal().sum().item()/confusion_matrix.sum(), (global_epoch+1)*global_steps_per_epoch)
    global_logger.add_scalar("validation/right-confidence", right_confidence/right_confidence_count if right_confidence_count else float("NaN"), (global_epoch+1)*global_steps_per_epoch)
    global_logger.add_scalar("validation/wrong-confidence", wrong_confidence/wrong_confidence_count if wrong_confidence_count else float("NaN"), (global_epoch+1)*global_steps_per_epoch)



def test(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_function: nn.Module)->None:
    global global_epochs, global_steps_per_epoch
    model.eval()
    test_loss_acc = 0.0
    confusion_matrix = torch.zeros((10,10), dtype=torch.int32)
    right_confidence = 0
    right_confidence_count = 0
    wrong_confidence = 0
    wrong_confidence_count = 0
    with torch.no_grad():
        for _, (test_x,test_y) in enumerate(dataloader):
            test_x, test_y = test_x.to("cuda"), test_y.to("cuda")
            test_z = model(test_x)
            confusion_matrix.put_(test_y.cpu().long()*10+test_z.cpu().argmax(1).long(), torch.ones((test_x.shape[0]), dtype=torch.int), accumulate=True)
            test_loss_acc += loss_function(test_z, test_y).item()
            with torch.no_grad():
                probabilities = test_z.softmax(dim=1)
                if probabilities[probabilities.argmax(1)==test_y].shape[0] > 0:
                    right_confidence += probabilities[probabilities.argmax(1)==test_y].max(dim=1)[0].mean()
                    right_confidence_count += 1
                if probabilities[probabilities.argmax(1)!=test_y].shape[0] > 0:
                    wrong_confidence += probabilities[probabilities.argmax(1)!=test_y].max(dim=1)[0].mean()
                    wrong_confidence_count += 1
    global_logger.add_scalar("test/loss", test_loss_acc/len(dataloader), (global_epoch+1)*global_steps_per_epoch)
    global_logger.add_scalar("test/accuracy", confusion_matrix.diagonal().sum()/confusion_matrix.sum(), (global_epoch+1)*global_steps_per_epoch)
    global_logger.add_scalar("test/right-confidence", right_confidence/right_confidence_count if right_confidence_count else float("NaN"), (global_epoch+1)*global_steps_per_epoch)
    global_logger.add_scalar("test/wrong-confidence", wrong_confidence/wrong_confidence_count if wrong_confidence_count else float("NaN"), (global_epoch+1)*global_steps_per_epoch)
    print("Test confusion matrix:", flush=True)
    print(confusion_matrix, flush=True)



##################Res-Net#####################
'''global_logger = SummaryWriter()
data = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])) #transforms.Resize((224, 224))
training_data, validation_data = torch.utils.data.random_split(data, [int(len(data)*0.75), int(len(data)*0.25)])
test_data = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])) #transforms.Resize((224, 224))
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True, worker_init_fn=random.seed(0)) #batch_size=64
validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=8, shuffle=True, worker_init_fn=random.seed(0)) #batch_size=64
testing_dataloader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True, worker_init_fn=random.seed(0)) #batch_size=64
model = ResNet18()
#model.load_state_dict(torch.load("net.pth"))
model.cuda();'''
################Simple-Net####################
global_logger = SummaryWriter()
data = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
training_data, validation_data = torch.utils.data.random_split(data, [int(len(data)*0.75), int(len(data)*0.25)])
test_data = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True, worker_init_fn=random.seed(seed))
validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True, worker_init_fn=random.seed(seed))
testing_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, worker_init_fn=random.seed(seed))
model = SimpleCIFARNet()
#model.load_state_dict(torch.load("net.pth"))
model.cuda()
##############################################

#####################Bilevel##########################
#inner_loss_function = LabelSmoothingLoss(10, torch.diag(torch.ones((10,))))
#inner_loss_function = LabelSmoothingLoss(10, torch.zeros((10,10)))
#inner_loss_function = LabelSmoothingLoss(10, torch.randn((10,10)))
inner_loss_function = LabelSmoothingLoss(10, torch.zeros((10,)))
#inner_loss_function = LabelSmoothingLoss(10, torch.randn((1,)))
inner_loss_function = inner_loss_function.cuda()
inner_optimizer = torch.optim.Adam(model.parameters())
outer_loss_function = nn.CrossEntropyLoss()
outer_optimizer = torch.optim.SGD(inner_loss_function.parameters(), lr=1e-1, momentum=0.9)
##################Nonbilevel##########################
'''inner_loss_function = nn.CrossEntropyLoss()
inner_optimizer = torch.optim.Adam(model.parameters())
outer_optimizer = None
outer_loss_function = nn.CrossEntropyLoss();'''
######################################################

#########Guarantee existence of inner_optimizer state########
_, (train_x, train_y) = next(enumerate(training_dataloader))
train_x, train_y = train_x.to("cuda"), train_y.to("cuda")
inner_optimizer.zero_grad()
inner_loss_function(model(train_x), train_y).backward()
inner_optimizer.step()
#############################################################

global_epochs = 7
global_epoch = 0
global_steps_per_epoch = min(10000,int(len(training_dataloader)))
global_step_in_epoch = 0
global_inner_iterations = 5

for epoch in range(global_epochs):
    global_epoch = epoch
    train(model, training_dataloader, inner_loss_function, inner_optimizer, validation_dataloader, outer_loss_function, outer_optimizer)
    #global_logger.add_image("smoothing-parameter", plot_matrix(inner_loss_function.smoothing.clone().detach().cpu().numpy()), global_epoch*global_steps_per_epoch+global_step_in_epoch)
    validate(model, validation_dataloader, outer_loss_function)
test(model, testing_dataloader, outer_loss_function)
global_logger.flush()
global_logger.close()
#torch.save(model.state_dict(), "net.pth")