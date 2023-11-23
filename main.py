import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time
from datetime import datetime
from model.MGSTGNN import Network
from model.discriminator import Discriminator_spatial,Discriminator_temporal

from trainer import Trainer
from utils.util import init_seed
from utils.dataloader import get_dataloader_pems
from utils.util import print_model_parameters
import warnings
warnings.filterwarnings('ignore')


#*************************************************************************#

from utils.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss
def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)

    return model
# Mode = 'train'
# DEBUG = 'True'
# DATASET = 'PEMSD3'      #PEMSD4 or PEMSD8
# DEVICE = 'cuda:0'
# MODEL = 'DDGCRN'

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default='PEMS07', type=str)
args.add_argument('--mode', default='train', type=str)
args.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
args.add_argument('--debug', default='False', type=eval)
args.add_argument('--model', default='MGSTGNN', type=str)
args.add_argument('--cuda', default=True, type=bool)
args1 = args.parse_args()

#get configuration
config_file = './config/{}.conf'.format(args1.dataset)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--in_steps', default=config['data']['in_steps'], type=int)
args.add_argument('--out_steps', default=config['data']['out_steps'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--adj_norm', default=config['data']['adj_norm'], type=eval)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--flow_dim', default=config['model']['flow_dim'], type=int)
args.add_argument('--period_dim', default=config['model']['period_dim'], type=int)
args.add_argument('--weekend_dim', default=config['model']['weekend_dim'], type=int)
args.add_argument('--holiday_dim', default=config['model']['holiday_dim'], type=int)
args.add_argument('--hop_dim', default=config['model']['hop_dim'], type=int)
args.add_argument('--weather_dim', default=config['model']['weather_dim'], type=int)
args.add_argument('--dim_discriminator', default=config['model']['dim_discriminator'], type=int)
args.add_argument('--alpha_discriminator', default=config['model']['alpha_discriminator'], type=float)
args.add_argument('--use_discriminator', default=config['model']['use_discriminator'], type=eval)

args.add_argument('--use_embs', default=config['model']['use_embs'], type=eval)
args.add_argument('--num_input_dim', default=config['model']['num_input_dim'], type=int)


args.add_argument('--input_embedding_dim', default=config['model']['input_embedding_dim'], type=int)
args.add_argument('--periods_embedding_dim', default=config['model']['periods_embedding_dim'], type=str)
args.add_argument('--weekend_embedding_dim', default=config['model']['weekend_embedding_dim'], type=int)
args.add_argument('--holiday_embedding_dim', default=config['model']['holiday_embedding_dim'], type=int)
args.add_argument('--spatial_embedding_dim', default=config['model']['spatial_embedding_dim'], type=int)
args.add_argument('--adaptive_embedding_dim', default=config['model']['adaptive_embedding_dim'], type=int)

args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_grus', default=config['model']['num_grus'], type=str)
args.add_argument('--periods', default=config['model']['periods'], type=str)
args.add_argument('--predict_time', default=config['model']['predict_time'], type=int)
args.add_argument('--use_back', default=config['model']['use_back'], type=eval)

#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--random', default=config['train']['random'], type=eval)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')

#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args = args.parse_args()
args.num_grus = [int(i) for i in list(args.num_grus.split(','))]
args.periods = [int(i) for i in list(args.periods.split(','))][:args.period_dim]
args.periods_embedding_dim = [int(i) for i in list(args.periods_embedding_dim.split(','))][:args.period_dim]

if args.random:
    args.seed = torch.randint(1000, (1,))
print(args)
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
model = Network(args)
model = model.to(args.device)
model = init_model(model)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader_pems(args1.dataset,args.batch_size,
                            args.val_ratio,args.test_ratio,args.in_steps,args.out_steps,
                            args.flow_dim,args.period_dim,args.weekend_dim,args.holiday_dim,
                            args.hop_dim,args.weather_dim)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss_generator = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss_generator = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss_generator = torch.nn.MSELoss().to(args.device)
elif args.loss_func == 'huber':
    loss_generator = torch.nn.HuberLoss().to(args.device)
else:
    raise ValueError
loss_discriminator = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler_G, lr_scheduler_spatial,lr_scheduler_temporal = None,None,None
discriminator_spatial,discriminator_temporal = None,None
optimizer_spatial, optimizer_temporal = None,None
if args.use_discriminator:
    discriminator_spatial = Discriminator_spatial(args)
    discriminator_spatial = discriminator_spatial.to(args.device)
    discriminator_spatial = init_model(discriminator_spatial)

    discriminator_temporal = Discriminator_temporal(args)
    discriminator_temporal = discriminator_temporal.to(args.device)
    discriminator_temporal = init_model(discriminator_temporal)
    optimizer_spatial = torch.optim.Adam(params=discriminator_spatial.parameters(),
                                       lr=args.lr_init*0.1,
                                       eps=1.0e-8,
                                       weight_decay=0,
                                       amsgrad=False)
    optimizer_temporal = torch.optim.Adam(params=discriminator_temporal.parameters(),
                                       lr=args.lr_init*0.1,
                                       eps=1.0e-8,
                                       weight_decay=0,
                                       amsgrad=False)
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_G,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)
    if args.use_discriminator:
        lr_scheduler_spatial = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_spatial,
                                                                     milestones=lr_decay_steps,
                                                                     gamma=args.lr_decay_rate)
        lr_scheduler_temporal = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_temporal,
                                                                     milestones=lr_decay_steps,
                                                                     gamma=args.lr_decay_rate)
#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'exps/logs', args.dataset, current_time)
args.log_dir = log_dir

#start training
trainer = Trainer(args,
                  model,discriminator_spatial,discriminator_temporal,
                  train_loader,val_loader,test_loader,scaler,
                  loss_generator,loss_discriminator,
                  optimizer_G,optimizer_spatial,optimizer_temporal,
                  lr_scheduler_G,lr_scheduler_spatial,lr_scheduler_temporal
                  )
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load('./pre-trained/{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError
