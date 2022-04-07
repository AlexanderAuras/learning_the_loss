import logging

import torch
import torchvision

import hydra
from omegaconf import DictConfig

from ltl.determinism_helper import seed_generators
seed_generators(0)

from ltl.simple_net import SimpleCIFARNet
from ltl.resnet18 import ResNet18
from ltl.label_smoothing_loss import LabelSmoothingLoss
from ltl.model import Model
from ltl.bilevel_model import BilevelModel
from ltl.transform_dataset import TransformDataset
from ltl import utils

logger = logging.getLogger(__name__)

@hydra.main(config_path="cifar/config", config_name="default.yaml")
def main(config: DictConfig) -> None:
    seed_generators(config.seed)
    logger.info("=======Initializing=======")
    logger.info("Loading datasets")
    if config.dataset.name == "cifar10":
        if config.model.name == "simple":
            data_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        elif config.model.name == "resnet18":
            data_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if config.label_noise_transition_matrix != None:
            label_transform = utils.label_gaussian_noise_transform(torch.tensor(config.label_noise_transition_matrix))
        else:
            label_transform = None
        data = torchvision.datasets.CIFAR10(root=f"{hydra.utils.get_original_cwd()}/cifar/data", train=True, download=True, transform=data_transform)
        training_data, validation_data = torch.utils.data.random_split(data, [len(data)-int(len(data)*(float(config.dataset.validation_percent)/100)), int(len(data)*(float(config.dataset.validation_percent)/100))])
        if config.train_noise:
            training_data = TransformDataset(training_data, None, label_transform)
        if config.validation_noise:
            validation_data = TransformDataset(validation_data, None, label_transform)
        test_data = torchvision.datasets.CIFAR10(root=f"{hydra.utils.get_original_cwd()}/cifar/data", train=False, download=True, transform=data_transform, target_transform=label_transform if config.test_noise else None)

    logger.info("Creating dataloaders")
    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=config.batch_size, shuffle=True, worker_init_fn=seed_generators(config.seed))
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=config.batch_size, shuffle=True, worker_init_fn=seed_generators(config.seed))
    testing_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=True, worker_init_fn=seed_generators(config.seed))

    logger.info("Creating module")
    if config.model.name == "simple":
        module = SimpleCIFARNet()
    elif config.model.name == "resnet18":
        module = ResNet18()

    if config.load_model_file != None:
        logger.info("Loading module from file")
        module.load_state_dict(torch.load(hydra.utils.get_original_cwd()+"/"+config.load_model_file))

    if config.cuda:
        logger.info("Moving module to cuda")
        module.cuda()

    logger.info("Creating loss functions")
    if config.bilevel:
        if config.inner_loss.name == "scalar_label_smoothing":
            if config.inner_loss.init == "random":
                inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, 10.0*torch.rand((1,))-5.0, config.inner_loss.sigmoid)
            elif config.inner_loss.init == "zero":
                inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, torch.zeros((1,)), config.inner_loss.sigmoid)
            elif config.inner_loss.init == "constant":
                inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, config.inner_loss.init_value*torch.ones((1,)), config.inner_loss.sigmoid)
        elif config.inner_loss.name == "vector_label_smoothing":
            if config.inner_loss.init == "random":
                inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, 10.0*torch.rand((config.dataset.class_count,))-5.0, config.inner_loss.sigmoid)
            if config.inner_loss.init == "zero":
                inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, torch.zeros((config.dataset.class_count,)), config.inner_loss.sigmoid)
            if config.inner_loss.init == "constant":
                inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, config.inner_loss.init_value*torch.ones((config.dataset.class_count,)), config.inner_loss.sigmoid)
        elif config.inner_loss.name == "matrix_label_smoothing":
            if config.inner_loss.init == "random":
                inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, torch.rand((config.dataset.class_count,config.dataset.class_count)), config.inner_loss.sigmoid)
            if config.inner_loss.init == "zero":
                inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, torch.zeros((config.dataset.class_count,config.dataset.class_count)), config.inner_loss.sigmoid)
            if config.inner_loss.init == "constant":
                inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, torch.full((config.dataset.class_count,config.dataset.class_count), config.inner_loss.init_value_other)+(config.inner_loss.init_value-config.inner_loss.init_value_other)*torch.diag(torch.ones((config.dataset.class_count,))), config.inner_loss.sigmoid)
    else:
        if config.inner_loss.name == "scalar_label_smoothing":
            inner_loss_function = LabelSmoothingLoss(config.dataset.class_count, config.inner_loss.init_value*torch.ones((1,)), config.inner_loss.sigmoid)
        elif config.inner_loss.name == "cross_entropy":
            inner_loss_function = torch.nn.CrossEntropyLoss()
    if config.cuda:
        logger.info("Moving loss function to cuda")
        inner_loss_function = inner_loss_function.cuda()
    outer_loss_function = torch.nn.CrossEntropyLoss()

    logger.info("Creating optimizers")
    if config.inner_optimizer.name == "adam":
        inner_optimizer = torch.optim.Adam(module.parameters(), lr=config.inner_optimizer.lr, betas=(config.inner_optimizer.beta1, config.inner_optimizer.beta2))
    elif config.inner_optimizer.name == "sgd":
        inner_optimizer = torch.optim.SGD(module.parameters(), lr=config.inner_optimizer.lr, momentum=config.inner_optimizer.momentum)
    if config.bilevel:
        if config.outer_optimizer.name == "adam":
            outer_optimizer = torch.optim.Adam(inner_loss_function.parameters(), lr=config.outer_optimizer.lr, betas=(config.outer_optimizer.beta1, config.outer_optimizer.beta2))
        elif config.outer_optimizer.name == "sgd":
            outer_optimizer = torch.optim.SGD(inner_loss_function.parameters(), lr=config.outer_optimizer.lr, momentum=config.outer_optimizer.momentum)
    else:
        outer_optimizer = None

    inner_lr_scheduler = None
    outer_lr_scheduler = None
    if config.inner_lr_scheduler.name != "none" or (config.bilevel and config.outer_lr_scheduler.name != "none"):
        logger.info("Creating learning rate schedulers")
        if config.inner_lr_scheduler.name != "none":
            if config.inner_lr_scheduler.name == "step":
                inner_lr_scheduler = torch.optim.lr_scheduler.StepLR(inner_optimizer, config.inner_lr_scheduler.step_size, config.inner_lr_scheduler.gamma)
        if (config.bilevel and config.outer_lr_scheduler.name != "none"):
            if config.outer_lr_scheduler.name == "step":
                outer_lr_scheduler = torch.optim.lr_scheduler.StepLR(outer_optimizer, config.outer_lr_scheduler.step_size, config.outer_lr_scheduler.gamma)

    logger.info("======Checking setup======")
    #########Guarantee existence of inner_optimizer state########
    logger.info("Get batch")
    _, (train_x, train_y) = next(enumerate(training_dataloader))
    if config.cuda:
        train_x, train_y = train_x.to("cuda"), train_y.to("cuda")
    inner_optimizer.zero_grad()
    logger.info("Forward")
    inner_loss_function(module(train_x), train_y).backward()
    logger.info("Backward")
    inner_optimizer.step()
    #############################################################

    if config.bilevel:
        model = BilevelModel(module, training_dataloader, validation_dataloader, testing_dataloader, inner_loss_function, inner_optimizer, config.inner_iterations, outer_loss_function, outer_optimizer, config.dataset.class_count, config.outer_batches_per_epoch, inner_lr_scheduler, outer_lr_scheduler)
    else:
        model = Model(module, training_dataloader, validation_dataloader, testing_dataloader, inner_loss_function, inner_optimizer, config.dataset.class_count, config.outer_batches_per_epoch, inner_lr_scheduler)

    logger.info("=========Training=========")
    model.validate(-1)
    for epoch in range(config.epochs):
        model.train(epoch)
        model.validate(epoch)
    model.test(config.epochs)

    if config.save_model_file != None:
        torch.save(module.state_dict(), hydra.utils.get_original_cwd()+"/"+config.save_model_file)
    model._tensorboard.flush()
    model._tensorboard.close()
    logger.info("===========Done===========")



if __name__ == "__main__":
    main()