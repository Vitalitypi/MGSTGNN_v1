import datetime

import torch
import torch.nn as nn
import math
import os
import time
import copy
import numpy as np
from utils.util import get_logger
from utils.metrics import All_Metrics


def record_loss(loss_file, loss):
    with open(loss_file, 'a') as f:
        line = "{:.4f}\n".format(loss)
        f.write(line) 


class Trainer(object):
    def __init__(self, 
                 args,
                 generator, discriminator, discriminator_spatial,discriminator_temporal,
                 train_loader, val_loader, test_loader, scaler,
                 loss_G, loss_D,
                 optimizer_G, optimizer_D, optimizer_spatial,optimizer_temporal,
                 lr_scheduler_G, lr_scheduler_D, lr_scheduler_spatial,lr_scheduler_temporal):

        super(Trainer, self).__init__()
        self.args = args
        self.num_nodes = args.num_nodes

        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_spatial = discriminator_spatial
        self.discriminator_temporal = discriminator_temporal
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_spatial = optimizer_spatial
        self.optimizer_temporal = optimizer_temporal
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.lr_scheduler_spatial = lr_scheduler_spatial
        self.lr_scheduler_temporal = lr_scheduler_temporal

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler

        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png') # when plot=True
        
        # log info
        if os.path.isdir(args.log_dir) == False:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info(f"Argument: {args}")
        for arg, value in sorted(vars(args).items()):
            self.logger.info(f"{arg}: {value}")

    def train_epoch(self, epoch):
        self.generator.train()
        total_loss_G = 0
        total_loss_D = 0
        total_loss_spatial = 0
        total_loss_temporal = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_size = data.shape[0]
            data = data[..., :self.args.input_dim] # [B'', W, N, 1]
            label = target[..., :self.args.output_dim]  # # [B'', H, N, 1]

            # Adversarial ground truths
            cuda = True if torch.cuda.is_available() else False
            TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            valid = torch.autograd.Variable(TensorFloat(batch_size*(self.args.in_steps + self.args.out_steps), 1).fill_(1.0), requires_grad=False)
            fake = torch.autograd.Variable(TensorFloat(batch_size*(self.args.in_steps + self.args.out_steps), 1).fill_(0.0), requires_grad=False)

            valid_spatial = torch.autograd.Variable(TensorFloat(batch_size*self.args.num_nodes, 1).fill_(1.0), requires_grad=False)
            fake_spatial = torch.autograd.Variable(TensorFloat(batch_size*self.args.num_nodes, 1).fill_(0.0), requires_grad=False)

            valid_temporal = torch.autograd.Variable(TensorFloat(batch_size*self.args.out_steps, 1).fill_(1.0), requires_grad=False)
            fake_temporal = torch.autograd.Variable(TensorFloat(batch_size*self.args.out_steps, 1).fill_(0.0), requires_grad=False)

            #-------------------------------------------------------------------
            # Train Generator 
            #-------------------------------------------------------------------
            self.optimizer_G.zero_grad()
                        
            # data and target shape: B, W, N, F, and B, H, N, F; output shape: B, H, N, F (F=1)
            output = self.generator(data)#, self.norm_dis_matrix
            data = data[..., :1]
            if self.args.real_value and self.args.dataset.lower() not in ['metr-la', 'pems-bay']: # it is depended on the output of model. If output is real data, the label should be reversed to real data
                label = self.scaler.inverse_transform(label)
            elif self.args.real_value and self.args.dataset.lower() in ['metr-la', 'pems-bay']:
                output = self.scaler.inverse_transform(output)  
            if self.args.dataset.lower() in ['metr-la', 'pems-bay']: 
                data = data[:, :, :, :1]
            # print(data.shape, output.shape, label.shape)
            fake_input = torch.cat((data, self.scaler.transform(output)), dim=1) if self.args.real_value else torch.cat((data, output), dim=1) # [B'', W, N, 1] // [B'', H, N, 1] -> [B'', W+H, N, 1]
            true_input = torch.cat((data, self.scaler.transform(label)), dim=1) if self.args.real_value else torch.cat((data, label), dim=1)
            fake_input_spatial = self.scaler.transform(output) if self.args.real_value else output # [B'', W, N, 1] // [B'', H, N, 1] -> [B'', W+H, N, 1]
            true_input_spatial = self.scaler.transform(label) if self.args.real_value else label

            fake_input_temporal = self.scaler.transform(output) if self.args.real_value else output # [B'', W, N, 1] // [B'', H, N, 1] -> [B'', W+H, N, 1]
            true_input_temporal = self.scaler.transform(label) if self.args.real_value else label

            loss_G = self.loss_G(output.cuda(), label) + 0.01 * self.loss_D(self.discriminator(fake_input), valid) + \
                     self.loss_D(self.discriminator_spatial(fake_input_spatial), valid_spatial) + \
                     self.loss_D(self.discriminator_temporal(fake_input_temporal), valid_temporal)
            loss_G.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm)
            self.optimizer_G.step()
            total_loss_G += loss_G.item()

            #-------------------------------------------------------------------
            # Train Discriminator 
            #-------------------------------------------------------------------
            self.optimizer_D.zero_grad()
            real_loss = self.loss_D(self.discriminator(true_input), valid)
            fake_loss = self.loss_D(self.discriminator(fake_input.detach()), fake)
            loss_D = 0.5 * (real_loss + fake_loss)
            loss_D.backward()
            self.optimizer_D.step()
            total_loss_D += loss_D.item()

            #-------------------------------------------------------------------
            # Train Discriminator_spatial
            #-------------------------------------------------------------------
            self.optimizer_spatial.zero_grad()
            real_loss_spatial = self.loss_D(self.discriminator_spatial(true_input_spatial), valid_spatial)
            fake_loss_spatial = self.loss_D(self.discriminator_spatial(fake_input_spatial.detach()), fake_spatial)
            loss_spatial = 0.5 * (real_loss_spatial + fake_loss_spatial)
            loss_spatial.backward()
            self.optimizer_spatial.step()
            total_loss_spatial += loss_spatial.item()

            #-------------------------------------------------------------------
            # Train Discriminator_temporal
            #-------------------------------------------------------------------
            self.optimizer_temporal.zero_grad()
            real_loss_temporal = self.loss_D(self.discriminator_temporal(true_input_temporal), valid_temporal)
            fake_loss_temporal = self.loss_D(self.discriminator_temporal(fake_input_temporal.detach()), fake_temporal)
            loss_temporal = 0.5 * (real_loss_temporal + fake_loss_temporal)
            loss_temporal.backward()
            self.optimizer_temporal.step()
            total_loss_temporal += loss_temporal.item()

            # log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Generator Loss: {:.6f} Pred Discriminator Loss: {:.6f} spatial Discriminator Loss: {:.6f} temporal Discriminator Loss: {:.6f}'.format(
                                 epoch, 
                                 batch_idx, self.train_per_epoch, 
                                 loss_G.item(), loss_D.item(), loss_spatial.item(), loss_temporal.item()))
        
        train_epoch_loss_G = total_loss_G / self.train_per_epoch # average generator loss
        train_epoch_loss_D = total_loss_D / self.train_per_epoch # average discriminator loss
        train_epoch_loss_spatial = total_loss_spatial / self.train_per_epoch # average discriminator loss
        train_epoch_loss_temporal = total_loss_temporal / self.train_per_epoch
        self.logger.info('**********Train Epoch {}: Averaged Generator Loss: {:.6f}, Averaged Pred Discriminator Loss: {:.6f}, Averaged spatial Discriminator Loss: {:.6f}, Averaged temporal Discriminator Loss: {:.6f}'.format(
                         epoch, 
                         train_epoch_loss_G,
                         train_epoch_loss_D,
                         train_epoch_loss_spatial,
                         train_epoch_loss_temporal
        ))

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
            self.lr_scheduler_spatial.step()
            self.lr_scheduler_temporal.step()
        return train_epoch_loss_G, train_epoch_loss_D, train_epoch_loss_spatial, train_epoch_loss_temporal

    def val_epoch(self, epoch, val_dataloader):
        self.generator.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim] # [B'', W, N, 1]
                label = target[..., :self.args.output_dim] # [B'', H, N, 1]
                output = self.generator(data)
                if self.args.real_value and self.args.dataset.lower() not in ['metr-la', 'pems-bay']:
                    label = self.scaler.inverse_transform(label)
                elif self.args.real_value and self.args.dataset.lower() in ['metr-la', 'pems-bay']:
                    output = self.scaler.inverse_transform(output)  
                loss = self.loss_G(output.cuda(), label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        
        return val_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list_G = []
        train_loss_list_D = []
        train_loss_list_spatial = []
        train_loss_list_temporal = []

        val_loss_list = []

        # loss file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        loss_dir = os.path.join(current_dir, 'exps/loss')
        if os.path.isdir(loss_dir) == False:
            os.makedirs(loss_dir, exist_ok=True)
        loss_file = './exps/loss/{}_{}_{}_val_loss.txt'.format(self.args.model, self.args.dataset,str(datetime.datetime.now()))
        if os.path.exists(loss_file):
            os.remove(loss_file)
            print('Recreate {}'.format(loss_file))

        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss_G, train_epoch_loss_D, train_epoch_loss_spatial, train_epoch_loss_temporal = self.train_epoch(epoch)
            
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            record_loss(loss_file, val_epoch_loss)

            train_loss_list_G.append(train_epoch_loss_G)
            train_loss_list_D.append(train_epoch_loss_D)
            train_loss_list_spatial.append(train_epoch_loss_spatial)
            train_loss_list_temporal.append(train_epoch_loss_temporal)
            val_loss_list.append(val_epoch_loss)

            if train_epoch_loss_G > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
                
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs! Training stops!".format(self.args.early_stop_patience))
                    # break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.generator.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        # if not self.args.debug:
        # test
        self.generator.load_state_dict(best_model)
        self.save_checkpoint()
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.generator, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.generator.state_dict(),
            'optimizer': self.optimizer_G.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(os.path.join(path, 'best_model.pth')) # path = args.log_dir
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim] # [B'', W, N, 1]
                label = target[..., :args.output_dim] # [B'', H, N, 1]
                output = model(data)
                y_true.append(label) # [B'', H, N, 1]
                y_pred.append(output) # [B'', H, N, 1]
        if args.real_value and args.dataset.lower() in ['metr-la', 'pems-bay']:
            y_true = torch.cat(y_true, dim=0)
        else:
            y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        
        if args.real_value and args.dataset.lower() not in ['metr-la', 'pems-bay']:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        
        # save predicted results as numpy format
        np.save(os.path.join(args.log_dir, '{}_true.npy'.format(args.dataset)), y_true.cpu().numpy())
        np.save(os.path.join(args.log_dir, '{}_pred.npy'.format(args.dataset)), y_pred.cpu().numpy())

        # each horizon point
        for t in range(y_true.shape[1]): # H
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape*100))
        # average all horizon point
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape*100))
