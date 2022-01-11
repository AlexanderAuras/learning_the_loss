import logging
import os

import torch
import torch.utils.tensorboard
import torchvision

import higher

import hydra
from omegaconf import DictConfig

from determinism_helper import seed_generators
seed_generators(0)

from resnet18 import ResNet18
from simple_net import SimpleCIFARNet
from label_smoothing_loss import LabelSmoothingLoss
import utils

logger = logging.getLogger(__name__)
tensorboard = None
config = None



def train(epoch:int,
            model: torch.nn.Module, 
            training_dataloader: torch.utils.data.DataLoader,
            inner_loss_function: torch.nn.Module, 
            inner_optimizer: torch.optim.Optimizer, 
            validation_dataloader: torch.utils.data.DataLoader, 
            outer_loss_function: torch.nn.Module, 
            outer_optimizer: torch.optim.Optimizer) -> None:
    global config, logger, tensorboard

    lastPercent = 0.0
    model.train()
    for batch in range(config.outer_batches_per_epoch):
        if (batch/float(config.outer_batches_per_epoch))*100.0 >= lastPercent+10.0:
            lastPercent += 10.0
            logger.info(f"\t{int(lastPercent)}%")
        if config.bilevel:
            bilevel_step(epoch, batch, model, training_dataloader, inner_loss_function, inner_optimizer, validation_dataloader, outer_loss_function, outer_optimizer)
        else:
            step(epoch, batch, model, training_dataloader, inner_loss_function, inner_optimizer)
    logger.info(f"\t100%")



def step(epoch:int,
            batch:int,
            model: torch.nn.Module,
            training_dataloader: torch.utils.data.DataLoader,
            loss_function: torch.nn.Module, 
            optimizer: torch.optim.Optimizer) -> None:
    global config, logger, tensorboard

    _, (train_x, train_y) = next(enumerate(training_dataloader))
    if config.cuda:
        train_x, train_y = train_x.to("cuda"), train_y.to("cuda")
    train_z = model(train_x)
    loss = loss_function(train_z, train_y)

    probabilities = train_z.softmax(dim=1)
    right_confidence = probabilities[probabilities.argmax(1)==train_y].max(dim=1)[0].mean().item()
    wrong_confidence = probabilities[probabilities.argmax(1)!=train_y].max(dim=1)[0].mean().item()
    tensorboard.add_scalar("train/inner-loss", loss.item(), epoch*config.outer_batches_per_epoch+batch)
    tensorboard.add_scalar("train/inner-right-confidence", right_confidence, epoch*config.outer_batches_per_epoch+batch)
    tensorboard.add_scalar("train/inner-wrong-confidence", wrong_confidence, epoch*config.outer_batches_per_epoch+batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def bilevel_step(epoch:int,
                    batch:int,
                    model: torch.nn.Module, 
                    training_dataloader: torch.utils.data.DataLoader,
                    inner_loss_function: torch.nn.Module, 
                    inner_optimizer: torch.optim.Optimizer, 
                    validation_dataloader: torch.utils.data.DataLoader, 
                    outer_loss_function: torch.nn.Module, 
                    outer_optimizer: torch.optim.Optimizer) -> None:
    global config, logger, tensorboard

    with higher.innerloop_ctx(model, inner_optimizer) as (fmodel, diffopt):
        ######Inner optimization#####
        inner_loss_acc = 0
        inner_right_confidence = 0
        inner_right_confidence_count = 0
        inner_wrong_confidence = 0
        inner_wrong_confidence_count = 0

        for i, (train_x, train_y) in enumerate(training_dataloader):
            if i >= config.inner_iterations:
                break

            if config.cuda:
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
        
        tensorboard.add_scalar("train/inner-loss", inner_loss_acc/config.inner_iterations, epoch*config.outer_batches_per_epoch+batch)
        tensorboard.add_scalar("train/inner-right-confidence", inner_right_confidence/inner_right_confidence_count if inner_right_confidence_count else float("NaN"), epoch*config.outer_batches_per_epoch+batch)
        tensorboard.add_scalar("train/inner-wrong-confidence", inner_wrong_confidence/inner_wrong_confidence_count if inner_wrong_confidence_count else float("NaN"), epoch*config.outer_batches_per_epoch+batch)
        
        new_model_state = fmodel.state_dict()
        new_optimizer_state = diffopt.state[0]
        #############################

        #######Outer optimization#####
        _, (val_x, val_y) = next(enumerate(validation_dataloader))
        if config.cuda:
            val_x, val_y = val_x.to("cuda"), val_y.to("cuda")
        val_z = fmodel(val_x)
        outer_loss = outer_loss_function(val_z, val_y)

        outer_probabilities = val_z.softmax(dim=1)
        tensorboard.add_scalar("train/outer-loss", outer_loss.item(), epoch*config.outer_batches_per_epoch+batch)
        tensorboard.add_scalar("train/outer-right-confidence", outer_probabilities[outer_probabilities.argmax(1)==val_y].max(dim=1)[0].mean().item(), epoch*config.outer_batches_per_epoch+batch)
        tensorboard.add_scalar("train/outer-wrong-confidence", outer_probabilities[outer_probabilities.argmax(1)!=val_y].max(dim=1)[0].mean().item(), epoch*config.outer_batches_per_epoch+batch)
        
        outer_optimizer.zero_grad()
        outer_loss.backward()
        outer_optimizer.step()
        
        if config.inner_loss.name == "scalar_label_smoothing":
            tensorboard.add_scalar("train/smoothing-parameter", inner_loss_function.smoothing.item(), epoch*config.outer_batches_per_epoch+batch)
        elif config.inner_loss.name == "vector_label_smoothing":
            tensorboard.add_figure("train/smoothing-parameter", utils.render_matrix(inner_loss_function.smoothing.clone().detach().unsqueeze(dim=0).cpu().numpy(), "Smoothing vector"), epoch*config.outer_batches_per_epoch+batch)
        elif config.inner_loss.name == "matrix_label_smoothing":
            tensorboard.add_figure("train/smoothing-parameter", utils.render_matrix(inner_loss_function.smoothing.clone().detach().cpu().numpy(), "Smoothing matrix"), epoch*config.outer_batches_per_epoch+batch)
        ##############################

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



def validate(epoch: int, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_function: torch.nn.Module) -> None:
    global config, logger, tensorboard
    model.eval()
    val_loss_acc = 0.0
    confusion_matrix = torch.zeros((10,10), dtype=torch.int32)
    right_confidence = 0
    right_confidence_count = 0
    wrong_confidence = 0
    wrong_confidence_count = 0
    with torch.no_grad():
        for _, (val_x, val_y) in enumerate(dataloader):
            if config.cuda:
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
    tensorboard.add_scalar("validation/loss", val_loss_acc/len(dataloader), (epoch+1)*config.outer_batches_per_epoch)
    tensorboard.add_scalar("validation/accuracy", confusion_matrix.diagonal().sum().item()/confusion_matrix.sum(), (epoch+1)*config.outer_batches_per_epoch)
    tensorboard.add_scalar("validation/right-confidence", right_confidence/right_confidence_count if right_confidence_count else float("NaN"), (epoch+1)*config.outer_batches_per_epoch)
    tensorboard.add_scalar("validation/wrong-confidence", wrong_confidence/wrong_confidence_count if wrong_confidence_count else float("NaN"), (epoch+1)*config.outer_batches_per_epoch)
    tensorboard.add_figure("validation/confusion-matrix", utils.render_matrix(torch.t(confusion_matrix).cpu().numpy(), "Confusion matrix"), (epoch+1)*config.outer_batches_per_epoch)



def test(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_function: torch.nn.Module) -> None:
    global config, logger, tensorboard
    model.eval()
    test_loss_acc = 0.0
    confusion_matrix = torch.zeros((10,10), dtype=torch.int32)
    right_confidence = 0
    right_confidence_count = 0
    wrong_confidence = 0
    wrong_confidence_count = 0
    with torch.no_grad():
        for _, (test_x,test_y) in enumerate(dataloader):
            if config.cuda:
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
    tensorboard.add_scalar("test/loss", test_loss_acc/len(dataloader), config.epochs*config.outer_batches_per_epoch)
    tensorboard.add_scalar("test/accuracy", confusion_matrix.diagonal().sum()/confusion_matrix.sum(), config.epochs*config.outer_batches_per_epoch)
    tensorboard.add_scalar("test/right-confidence", right_confidence/right_confidence_count if right_confidence_count else float("NaN"), config.epochs*config.outer_batches_per_epoch)
    tensorboard.add_scalar("test/wrong-confidence", wrong_confidence/wrong_confidence_count if wrong_confidence_count else float("NaN"), config.epochs*config.outer_batches_per_epoch)
    tensorboard.add_figure("test/confusion-matrix", utils.render_matrix(torch.t(confusion_matrix).cpu().numpy(), "Confusion matrix"), config.epochs*config.outer_batches_per_epoch)
    logger.info("Test confusion matrix:")
    logger.info("\n"+str(confusion_matrix))



#TODO Fix smoothing vector?
#TODO Seperate into files?
#TODO Test Label-Noise, test nonuniform label distributions
#TODO Latex
@hydra.main(config_path="config", config_name="default.yaml")
def main(cfg: DictConfig) -> None:
    global config, logger, tensorboard
    config = cfg
    tensorboard = torch.utils.tensorboard.SummaryWriter(".")

    logger.info("=======Initializing=======")
    logger.info("Loading datasets")
    if config.dataset.name == "cifar10":
        if config.model.name == "simple":
            data_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        elif config.model.name == "resnet18":
            data_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        data = torchvision.datasets.CIFAR10(root=f"{hydra.utils.get_original_cwd()}/data", train=True, download=True, transform=data_transform)
        training_data, validation_data = torch.utils.data.random_split(data, [len(data)-int(len(data)*(float(config.dataset.validation_percent)/100)), int(len(data)*(float(config.dataset.validation_percent)/100))])
        test_data = torchvision.datasets.CIFAR10(root=f"{hydra.utils.get_original_cwd()}/data", train=False, download=True, transform=data_transform)

    logger.info("Creating dataloaders")
    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=config.model.batch_size, shuffle=True, worker_init_fn=seed_generators(config.seed))
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=config.model.batch_size, shuffle=True, worker_init_fn=seed_generators(config.seed))
    testing_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.model.batch_size, shuffle=True, worker_init_fn=seed_generators(config.seed))

    logger.info("Creating model")
    if config.model.name == "simple":
        model = SimpleCIFARNet()
    elif config.model.name == "resnet18":
        model = ResNet18()

    if config.load_model_file != None:
        logger.info("Loading model from file")
        model.load_state_dict(torch.load(config.load_model_file))

    if config.cuda:
        logger.info("Moving model to cuda")
        model.cuda()

    logger.info("Creating loss functions")
    if config.bilevel:
        if config.inner_loss.name == "scalar_label_smoothing":
            inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, 0.5*torch.randn((1,)))
        elif config.inner_loss.name == "vector_label_smoothing":
            inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, 0.5*torch.randn((config.dataset.class_count,)))
        elif config.inner_loss.name == "matrix_label_smoothing":
            inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, 0.5*torch.randn((config.dataset.class_count,config.dataset.class_count)))
    else:
        inner_loss_function = torch.nn.CrossEntropyLoss()
    if config.cuda:
        logger.info("Moving loss function to cuda")
        inner_loss_function = inner_loss_function.cuda()
    outer_loss_function = torch.nn.CrossEntropyLoss()

    logger.info("Creating optimizers")
    if config.inner_optimizer.name == "adam":
        inner_optimizer = torch.optim.Adam(model.parameters(), lr=config.inner_optimizer.lr)
    elif config.inner_optimizer.name == "sgd":
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=config.inner_optimizer.lr)
    if config.bilevel:
        if config.outer_optimizer.name == "adam":
            outer_optimizer = torch.optim.Adam(inner_loss_function.parameters(), lr=config.outer_optimizer.lr)
        elif config.outer_optimizer.name == "sgd":
            outer_optimizer = torch.optim.SGD(inner_loss_function.parameters(), lr=config.outer_optimizer.lr)
    else:
        outer_optimizer = None

    logger.info("======Doing test step=====")
    #########Guarantee existence of inner_optimizer state########
    logger.info("Get batch")
    _, (train_x, train_y) = next(enumerate(training_dataloader))
    if config.cuda:
        train_x, train_y = train_x.to("cuda"), train_y.to("cuda")
    inner_optimizer.zero_grad()
    logger.info("Forward")
    inner_loss_function(model(train_x), train_y).backward()
    logger.info("Backward")
    inner_optimizer.step()
    #############################################################

    logger.info("=========Training=========")
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch+1}:")
        train(epoch, model, training_dataloader, inner_loss_function, inner_optimizer, validation_dataloader, outer_loss_function, outer_optimizer)
        validate(epoch, model, validation_dataloader, outer_loss_function)
    test(model, testing_dataloader, outer_loss_function)

    if config.save_model_file != None:
        torch.save(model.state_dict(), config.save_model_file)
    tensorboard.flush()
    tensorboard.close()
    logger.info("===========Done===========")



if __name__ == "__main__":
    main()