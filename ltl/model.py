import logging

import torch
import torch.utils.tensorboard
import torchmetrics

from ltl import utils



class Model:
    def __init__(self, module, training_dataloader, validation_dataloader, testing_dataloader, loss_function, optimizer, class_count, outer_batches_per_epoch, lr_scheduler):
        self._logger = logging.getLogger(__name__)
        self._tensorboard = torch.utils.tensorboard.SummaryWriter(".")
        self._training_dataloader = training_dataloader
        self._validation_dataloader = validation_dataloader
        self._testing_dataloader = testing_dataloader
        self._module = module
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._class_count = class_count
        self._outer_batches_per_epoch = outer_batches_per_epoch
        self._lr_scheduler = lr_scheduler



    def train(self, epoch):
        self._module.train()
        lastPercent = 0.0

        self._logger.info(f"Epoch {epoch+1}:")
        self._logger.info(f"\t0%")
        batch = 0
        while batch < self._outer_batches_per_epoch:
            for x, y in self._training_dataloader:
                if batch == self._outer_batches_per_epoch:
                    break
                if (batch/float(self._outer_batches_per_epoch))*100.0 >= lastPercent+10.0:
                    lastPercent += 10.0
                    self._logger.info(f"\t{int(lastPercent)}%")

                x, y = x.to(next(self._module.parameters()).device), y.to(next(self._module.parameters()).device)
                z = self._module(x)
                loss = self._loss_function(z, y)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                probabilities = z.softmax(dim=1)
                right_confidence = probabilities[probabilities.argmax(1)==y].max(dim=1)[0].mean().item()
                wrong_confidence = probabilities[probabilities.argmax(1)!=y].max(dim=1)[0].mean().item()
                self._tensorboard.add_scalar("train/inner-accuracy",         (probabilities.argmax(1)==y).sum()/y.shape[0], epoch*self._outer_batches_per_epoch+batch)
                self._tensorboard.add_scalar("train/inner-loss",             loss.item(),                                   epoch*self._outer_batches_per_epoch+batch)
                self._tensorboard.add_scalar("train/inner-right-confidence", right_confidence,                              epoch*self._outer_batches_per_epoch+batch)
                self._tensorboard.add_scalar("train/inner-wrong-confidence", wrong_confidence,                              epoch*self._outer_batches_per_epoch+batch)

                batch += 1

        if self._lr_scheduler != None:
            self._lr_scheduler.step()

        self._logger.info("\t100%")
        self._logger.info("")



    def validate(self, epoch):
        self._module.eval()

        loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore").to(next(self._module.parameters()).device)
        accuracy_metric = torchmetrics.Accuracy().to(next(self._module.parameters()).device)
        confusion_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=self._class_count)
        right_confidence_metric = torchmetrics.MeanMetric(nan_strategy="ignore").to(next(self._module.parameters()).device)
        wrong_confidence_metric = torchmetrics.MeanMetric(nan_strategy="ignore").to(next(self._module.parameters()).device)

        with torch.no_grad():
            for _, (x, y) in enumerate(self._validation_dataloader):
                x, y = x.to(next(self._module.parameters()).device), y.to(next(self._module.parameters()).device)
                z = self._module(x)
                loss_metric.update(self._loss_function(z, y).item())
                probabilities = z.softmax(dim=1)
                accuracy_metric.update(probabilities, y)
                confusion_matrix_metric.update(probabilities.cpu(), y.cpu())
                if probabilities[probabilities.argmax(1)==y].shape[0] > 0:
                    right_confidence_metric.update(probabilities[probabilities.argmax(1)==y].max(dim=1)[0].mean().item())
                if probabilities[probabilities.argmax(1)!=y].shape[0] > 0:
                    wrong_confidence_metric.update(probabilities[probabilities.argmax(1)!=y].max(dim=1)[0].mean().item())
        
        self._tensorboard.add_scalar("validation/loss",             loss_metric.compute().item(),             (epoch+1)*self._outer_batches_per_epoch)
        self._tensorboard.add_scalar("validation/accuracy",         accuracy_metric.compute().item(),         (epoch+1)*self._outer_batches_per_epoch)
        self._tensorboard.add_scalar("validation/right-confidence", right_confidence_metric.compute().item(), (epoch+1)*self._outer_batches_per_epoch)
        self._tensorboard.add_scalar("validation/wrong-confidence", wrong_confidence_metric.compute().item(), (epoch+1)*self._outer_batches_per_epoch)
        self._tensorboard.add_figure("validation/confusion-matrix", utils.render_matrix(torch.t(confusion_matrix_metric.compute()).to(torch.int32).numpy(), "Confusion matrix", "Predicted", "Actual"), (epoch+1)*self._outer_batches_per_epoch)
        self._logger.info("Validation loss: "+str(loss_metric.compute().item()))
        self._logger.info("Validation accuracy: "+str(accuracy_metric.compute().item()))
        self._logger.info("")
        self._logger.info("")


    def test(self, epoch):
        self._module.eval()

        loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore").to(next(self._module.parameters()).device)
        accuracy_metric = torchmetrics.Accuracy().to(next(self._module.parameters()).device)
        confusion_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=10)
        right_confidence_metric = torchmetrics.MeanMetric(nan_strategy="ignore").to(next(self._module.parameters()).device)
        wrong_confidence_metric = torchmetrics.MeanMetric(nan_strategy="ignore").to(next(self._module.parameters()).device)

        with torch.no_grad():
            for _, (x, y) in enumerate(self._testing_dataloader):
                x, y = x.to(next(self._module.parameters()).device), y.to(next(self._module.parameters()).device)
                z = self._module(x)
                loss_metric.update(self._loss_function(z, y).item())
                probabilities = z.softmax(dim=1)
                accuracy_metric.update(probabilities, y)
                confusion_matrix_metric.update(probabilities.cpu(), y.cpu())
                if probabilities[probabilities.argmax(1)==y].shape[0] > 0:
                    right_confidence_metric.update(probabilities[probabilities.argmax(1)==y].max(dim=1)[0].mean().item())
                if probabilities[probabilities.argmax(1)!=y].shape[0] > 0:
                    wrong_confidence_metric.update(probabilities[probabilities.argmax(1)!=y].max(dim=1)[0].mean().item())
        
        self._tensorboard.add_scalar("test/loss",             loss_metric.compute().item(),             epoch*self._outer_batches_per_epoch)
        self._tensorboard.add_scalar("test/accuracy",         accuracy_metric.compute().item(),         epoch*self._outer_batches_per_epoch)
        self._tensorboard.add_scalar("test/right-confidence", right_confidence_metric.compute().item(), epoch*self._outer_batches_per_epoch)
        self._tensorboard.add_scalar("test/wrong-confidence", wrong_confidence_metric.compute().item(), epoch*self._outer_batches_per_epoch)
        self._tensorboard.add_figure("test/confusion-matrix", utils.render_matrix(torch.t(confusion_matrix_metric.compute()).to(torch.int32).numpy(), "Confusion matrix", "Predicted", "Actual"), epoch*self._outer_batches_per_epoch)
        self._logger.info("Test confusion matrix:\n"+str(confusion_matrix_metric.compute()))
        self._logger.info("Test loss: "+str(loss_metric.compute().item()))
        self._logger.info("Test accuracy: "+str(accuracy_metric.compute().item()))
        self._logger.info("")
        self._logger.info("")