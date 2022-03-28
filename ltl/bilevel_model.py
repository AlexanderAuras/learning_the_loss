import logging

import torch

import torchmetrics

import higher

from ltl import utils

from ltl.model import Model



class BilevelModel(Model):
    def __init__(self, _module, training_dataloader, validation_dataloader, testing_dataloader, inner_loss_function, inner_optimizer, inner_iterations, outer_loss_function, outer_optimizer, class_count, outer_batches_per_epoch, inner_lr_scheduler, outer_lr_scheduler):
        super().__init__(_module, training_dataloader, validation_dataloader, testing_dataloader, inner_loss_function, None, class_count, outer_batches_per_epoch, None)
        self._logger = logging.getLogger(__name__)
        self._inner_loss_function = inner_loss_function
        self._inner_optimizer = inner_optimizer
        self._outer_loss_function = outer_loss_function
        self._outer_optimizer = outer_optimizer
        self._inner_iterations = inner_iterations
        self._inner_lr_scheduler = inner_lr_scheduler
        self._outer_lr_scheduler = outer_lr_scheduler



    def train(self, epoch):
        self._module.train()
        lastPercent = 0.0
        self._logger.info(f"Epoch {epoch+1}:")

        inner_iter = iter(self._training_dataloader)
        inner_remaining = len(self._training_dataloader)
        outer_iter = iter(self._validation_dataloader)
        outer_remaining = len(self._validation_dataloader)

        for outer_batch in range(self._outer_batches_per_epoch):
            if (outer_batch/float(self._outer_batches_per_epoch))*100.0 >= lastPercent+10.0:
                lastPercent += 10.0
                self._logger.info(f"\t{int(lastPercent)}%")
            
            inner_loss_metric = torchmetrics.MeanMetric(nan_strategy="ignore").to(next(self._module.parameters()).device)
            inner_right_confidence_metric = torchmetrics.MeanMetric(nan_strategy="ignore").to(next(self._module.parameters()).device)
            inner_wrong_confidence_metric = torchmetrics.MeanMetric(nan_strategy="ignore").to(next(self._module.parameters()).device)
            inner_accuracy_metric = torchmetrics.Accuracy().to(next(self._module.parameters()).device)

            with higher.innerloop_ctx(self._module, self._inner_optimizer) as (fmodule, diffopt):
                ######Inner optimization#####
                for _ in range(self._inner_iterations):
                    if inner_remaining == 0:
                        inner_iter = iter(self._training_dataloader)
                        inner_remaining = len(self._training_dataloader)
                    x, y = next(inner_iter)
                    inner_remaining -= 1
                    x, y = x.to(next(self._module.parameters()).device), y.to(next(self._module.parameters()).device)
                    z = fmodule(x)
                    inner_loss = self._inner_loss_function(z, y)
                    diffopt.step(inner_loss)
                    
                    inner_loss_metric.update(inner_loss.item())
                    probabilities = z.softmax(dim=1)
                    inner_accuracy_metric.update(probabilities, y)
                    if probabilities[probabilities.argmax(1)==y].shape[0] > 0:
                        inner_right_confidence_metric.update(probabilities[probabilities.argmax(1)==y].max(dim=1)[0].mean().item())
                    if probabilities[probabilities.argmax(1)!=y].shape[0] > 0:
                        inner_wrong_confidence_metric.update(probabilities[probabilities.argmax(1)!=y].max(dim=1)[0].mean().item())

                self._tensorboard.add_scalar("train/inner-accuracy",         inner_accuracy_metric.compute().item(),         epoch*self._outer_batches_per_epoch+outer_batch)
                self._tensorboard.add_scalar("train/inner-loss",             inner_loss_metric.compute().item(),             epoch*self._outer_batches_per_epoch+outer_batch)
                self._tensorboard.add_scalar("train/inner-right-confidence", inner_right_confidence_metric.compute().item(), epoch*self._outer_batches_per_epoch+outer_batch)
                self._tensorboard.add_scalar("train/inner-wrong-confidence", inner_wrong_confidence_metric.compute().item(), epoch*self._outer_batches_per_epoch+outer_batch)
                
                new_module_state = fmodule.state_dict()
                new_optimizer_state = diffopt.state[0]
                #############################

                #######Outer optimization#####
                if outer_remaining == 0:
                    outer_iter = iter(self._validation_dataloader)
                    outer_remaining = len(self._validation_dataloader)
                x, y = next(outer_iter)
                outer_remaining -= 1
                x, y = x.to(next(self._module.parameters()).device), y.to(next(self._module.parameters()).device)
                z = fmodule(x)
                outer_loss = self._outer_loss_function(z, y)

                probabilities = z.softmax(dim=1)
                self._tensorboard.add_scalar("train/outer-accuracy",         (probabilities.argmax(1)==y).sum()/y.shape[0],                         epoch*self._outer_batches_per_epoch+outer_batch)
                self._tensorboard.add_scalar("train/outer-loss",             outer_loss.item(),                                                     epoch*self._outer_batches_per_epoch+outer_batch)
                self._tensorboard.add_scalar("train/outer-right-confidence", probabilities[probabilities.argmax(1)==y].max(dim=1)[0].mean().item(), epoch*self._outer_batches_per_epoch+outer_batch)
                self._tensorboard.add_scalar("train/outer-wrong-confidence", probabilities[probabilities.argmax(1)!=y].max(dim=1)[0].mean().item(), epoch*self._outer_batches_per_epoch+outer_batch)
                
                self._outer_optimizer.zero_grad()
                outer_loss.backward()
                self._outer_optimizer.step()
                
                if self._inner_loss_function.smoothing.shape[0] == 1:
                    self._tensorboard.add_scalar("train/smoothing-parameter", self._inner_loss_function.smoothing.item(), epoch*self._outer_batches_per_epoch+outer_batch)
                elif self._inner_loss_function.smoothing.shape[0] != 1 and (len(self._inner_loss_function.smoothing.shape) == 1 or self._inner_loss_function.smoothing.shape[1] == 1):
                    self._tensorboard.add_figure("train/smoothing-parameter", utils.render_matrix(self._inner_loss_function.smoothing.clone().detach().unsqueeze(dim=0).cpu().numpy(), "Smoothing vector", "", "Weight"), epoch*self._outer_batches_per_epoch+outer_batch)
                elif self._inner_loss_function.smoothing.shape[0] != 1 and self._inner_loss_function.smoothing.shape[1] != 1:
                    self._tensorboard.add_figure("train/smoothing-parameter", utils.render_matrix(utils.translate_to_zero(self._inner_loss_function.smoothing.clone().detach().cpu().numpy()), "Smoothing matrix", "Correct label", "Weight"), epoch*self._outer_batches_per_epoch+outer_batch)
                ##############################

            utils.copy_higher_to_torch(new_module_state, new_optimizer_state, self._module, self._inner_optimizer)

        if self._inner_lr_scheduler != None:
            self._inner_lr_scheduler.step()
        if self._outer_lr_scheduler != None:
            self._outer_lr_scheduler.step()

        torch.cuda.empty_cache()
        
        self._logger.info("\t100%")
        self._logger.info("")