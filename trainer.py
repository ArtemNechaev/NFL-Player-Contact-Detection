from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, Dataset
from IPython.display import clear_output
import matplotlib.pyplot as plt
import inspect

import numpy as np
import os
import gc

class Trainer():
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, model, 
                 batch_size, num_epoch, loss_fn, grad_acc_step=1, max_grad_norm=1,  eval_batch_size=None,
                 max_eval_batches=None, 
                 lr=5e-5, weight_decay=0,
                 train_transform = None,
                 val_transform = None,
                 optimizer=None, scheduler=None,
                 metrics: List[Tuple[str, Callable]] = [],
                 eval_freq=5000, saving_path='.', model_name='model', sampler=None, 
                 freq_online_loss_plot: int = -1):
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.batch_size = batch_size
        self.device = model.device
        self.num_epoch = num_epoch
        self.metrics = metrics
        self.eval_freq = eval_freq
        self.saving_path = saving_path
        self.model_name = model_name + '.pt'
        self.max_eval_batches = max_eval_batches
        self.grad_acc_step = grad_acc_step
        self.max_grad_norm = max_grad_norm
        self.freq_online_loss_plot = freq_online_loss_plot
        
        self.train_transform = train_transform.to(self.device)
        self.val_transform = val_transform.to(self.device)
        
        if eval_batch_size is None:
            self.eval_batch_size = batch_size
        else:
            self.eval_batch_size = eval_batch_size

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if optimizer is not None:
            self.optimizer = optimizer(
                optimizer_grouped_parameters, lr=lr,)
        else:
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=lr,)
            
        self.scheduler = scheduler

        if sampler is None:
            self.sampler = RandomSampler(self.train_dataset)
        else:
            self.sampler = sampler
        #weight=torch.tensor([0.5,1,2]
        #weight=torch.tensor([1,1,1])
        self.loss_fn = loss_fn
            
    def data2device(self, data: Dict):
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items() }


    def compute_loss(self, batch, **kwargs):
        
        outputs = self.model(**batch)
        return self.loss_fn(outputs, **batch), outputs
    
    def evaluation(self, val_dataloader=None, max_eval_batches=None, ):
        if val_dataloader is None:

            val_dataloader = DataLoader(self.val_dataset, 
                                        batch_size=self.eval_batch_size, 
                                        collate_fn=self.val_dataset.collate_fn, 
                                        shuffle=False)

        val_loss = 0
        eval_outputs = {"preds": [], "targets": []}
        if max_eval_batches is None:
            max_eval_batches = len(val_dataloader)
            
        self.model.eval()
        for eval_step, val_batch in enumerate(val_dataloader):
            if eval_step > max_eval_batches:
                break

            val_batch = self.data2device(val_batch)
            if self.val_transform is not None:
                val_batch = self.val_transform(val_batch)
                
            with torch.no_grad():

                loss, logits = self.compute_loss(val_batch)
                val_loss += loss.item()
                
                ids = val_batch['label'].flatten() != -100
                y_pred = logits.argmax(-1).flatten()[ids]
                y_true = val_batch['label'].flatten()[ids]
                
                eval_outputs['preds'].extend(y_pred.cpu().numpy())
                eval_outputs['targets'].extend(y_true.cpu().numpy())
                
        val_loss = val_loss/max_eval_batches
        return val_loss, eval_outputs
    
    def train(self):
       try:
          best_val_loss = float('inf')

          train_dataloader = DataLoader(self.train_dataset, 
                                        batch_size=self.batch_size, 
                                        collate_fn = self.train_dataset.collate_fn,
                                        sampler=self.sampler)

          val_dataloader = DataLoader(self.val_dataset, 
                                      batch_size=self.eval_batch_size, 
                                      collate_fn = self.val_dataset.collate_fn,
                                      shuffle=False)

          epoch_loss_set = {'train': [], 'val': []}
          metrics_values = {}

          if self.max_eval_batches is None:
              self.max_eval_batches = len(val_dataloader)


          for epoch in range(self.num_epoch):
              self.epoch = epoch

              train_loss_set, val_loss_set, x_val_set  = [], [], []
              train_loss = 0

              for step, batch in enumerate(train_dataloader):

                  # Training
                  self.step = step
                  self.model.train()
                  batch = self.data2device(batch)
                  if self.train_transform is not None:
                      batch = self.train_transform(batch)
                    
                  loss, pred_scores = self.compute_loss(batch)
                  loss.backward()
                  if (step + 1) % self.grad_acc_step == 0:
                      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                      self.optimizer.step()
                      self.optimizer.zero_grad()
                  train_loss_set.append(loss.item())     
                  train_loss += loss.item()

                  # Evaluation
                  if step != 0 and (step % self.eval_freq == 0 or step == len(train_dataloader)-1):

                      val_loss, eval_outputs = self.evaluation(val_dataloader, self.max_eval_batches)
                      val_loss_set.append(val_loss)
                      x_val_set.append(step)

                      # log metrics and loss to tensorboard
                      for name, func in self.metrics:
                          value = func(**eval_outputs)

                          if name not in metrics_values:
                              metrics_values[name] = {'batch': [], 'epoch': []}
                          metrics_values[name]['batch'].append(value)


                      # savemodel
                      if val_loss < best_val_loss:
                          best_val_loss = val_loss
                          self.model.save_pretrained(self.saving_path)


                      if self.scheduler is not None:
                          self.scheduler.step(val_loss)

                  # plot online loss
                  if self.freq_online_loss_plot > 0 and step % self.freq_online_loss_plot == 0:
                      clear_output(True)

                      fig, axs  = plt.subplots(len(self.metrics) + 1, 2,figsize=(18, 5 + 5*len(self.metrics)))
                      if len(axs.shape) == 1:
                          axs = axs[None, :]
                      batch_plot, epoch_plot = axs[0, 0], axs[0, 1]

                      batch_plot.plot(train_loss_set)
                      batch_plot.plot(x_val_set, val_loss_set)

                      batch_plot.set_title("Training/Val loss")


                      for k, v in epoch_loss_set.items():
                          epoch_plot.plot(v, label=k)

                      for i, (k, v) in enumerate(metrics_values.items()):
                          axs[i+1,0].set_title(k)
                          axs[i+1,0].plot(x_val_set, v['batch'], c='orange')
                          axs[i+1,1].plot(v['epoch'], c='orange')

                      for i, ax in enumerate(axs.flatten()):
                          ax.grid()
                          ax.legend()
                          if i//2 == 0:
                              ax.set_xlabel("Batch")
                          else:
                              ax.set_xlabel("Epoch")

                      plt.show()

              #add values to epoch plot
              epoch_loss_set['train'].append(train_loss / len(train_dataloader))
              epoch_loss_set['val'].append(val_loss)

              for k in metrics_values.keys():
                  metrics_values[k]['epoch'].append(metrics_values[k]['batch'][-1])
                  metrics_values[k]['batch'] = []

          print("Train Loss: {0:.5f}".format(train_loss / len(train_dataloader)))
          print("Validation Loss: {0:.5f}".format(val_loss))
        
       except KeyboardInterrupt:
          print('KeyboardInterrupt')
        
