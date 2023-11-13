import os
import sys
import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime

import yaml
import torch.nn.functional as F
from model.MGSTGNN import Network
from model.discriminator import Discriminator, Discriminator_spatial, Discriminator_temporal
# from trainer import Trainer
from trainer import Trainer
from utils.dataloader import get_dataloader_pems, get_adj_dis_matrix, norm_adj, get_dataloader_meta_la, construct_adj

# Correlated training configuration
from utils.util import get_device, print_model_parameters, masked_mae_loss

Mode = 'Train'
DATASET = 'PEMS04'
MODEL = "MGSTGNN"
DEBUG = 'True'
ADJ_FILE = './dataset/{}/{}.csv'.format(DATASET, DATASET)
# get configuration
config_file = './config/{}.yaml'.format(DATASET)
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

def get_arguments():
    # parser
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--mode', default=Mode, type=str)
    parser.add_argument('--debug', default=DEBUG, type=eval)
    parser.add_argument('--model', default=MODEL, type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    # data
    parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    parser.add_argument('--in_steps', default=config['data']['in_steps'], type=int)
    parser.add_argument('--out_steps', default=config['data']['out_steps'], type=int)
    parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    parser.add_argument('--adj_norm', default=config['data']['adj_norm'], type=eval)
    # model
    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--flow_dim', default=config['model']['flow_dim'], type=int)
    parser.add_argument('--period_dim', default=config['model']['period_dim'], type=int)
    parser.add_argument('--weekend_dim', default=config['model']['weekend_dim'], type=int)
    parser.add_argument('--holiday_dim', default=config['model']['holiday_dim'], type=int)
    parser.add_argument('--hop_dim', default=config['model']['hop_dim'], type=int)
    parser.add_argument('--weather_dim', default=config['model']['weather_dim'], type=int)
    parser.add_argument('--dim_discriminator', default=config['model']['dim_discriminator'], type=int)
    parser.add_argument('--alpha_discriminator', default=config['model']['alpha_discriminator'], type=float)

    parser.add_argument('--dim_embed_feature', default=config['model']['dim_embed_feature'], type=int)
    parser.add_argument('--input_embedding_dim', default=config['model']['input_embedding_dim'], type=int)
    parser.add_argument('--periods_embedding_dim', default=config['model']['periods_embedding_dim'], type=list)
    parser.add_argument('--weekend_embedding_dim', default=config['model']['weekend_embedding_dim'], type=int)
    parser.add_argument('--holiday_embedding_dim', default=config['model']['holiday_embedding_dim'], type=int)
    parser.add_argument('--spatial_embedding_dim', default=config['model']['spatial_embedding_dim'], type=int)
    parser.add_argument('--adaptive_embedding_dim', default=config['model']['adaptive_embedding_dim'], type=int)

    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--periods', default=config['model']['periods'][:config['model']['period_dim']], type=list)
    parser.add_argument('--predict_time', default=config['model']['predict_time'], type=int)
    parser.add_argument('--gat_hidden', default=config['model']['gat_hidden'], type=int)
    parser.add_argument('--mlp_hidden', default=config['model']['mlp_hidden'], type=int)
    parser.add_argument('--gat_drop', default=config['model']['gat_drop'], type=float)
    parser.add_argument('--gat_heads', default=config['model']['gat_heads'], type=int)
    parser.add_argument('--gat_alpha', default=config['model']['gat_alpha'], type=float)
    parser.add_argument('--gat_concat', default=config['model']['gat_concat'], type=eval)
    parser.add_argument('--mlp_act', default=config['model']['mlp_act'], type=str)
    parser.add_argument('--mlp_drop', default=config['model']['mlp_drop'], type=float)
    parser.add_argument('--num_gat', default=config['model']['num_gat'], type=int)
    parser.add_argument('--num_back', default=config['model']['num_back'], type=int)
    # train
    parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--random', default=config['train']['random'], type=eval)
    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
    # test
    parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
    # log
    parser.add_argument('--log_dir', default='./exps/logs/', type=str)
    parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
    parser.add_argument('--plot', default=config['log']['plot'], type=eval)
    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")

    return args

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)

    return model

if __name__ == "__main__":
    args = get_arguments()
    if args.random:
        args.seed = torch.randint(1000, (1,)) # set random seed here
    print("seed:",args.seed)
    args.device = get_device(args)

    # get norm adj_matrix, norm dis_matrix
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    st_adj,st_dis = None,None
    if args.dataset.lower() == 'pems03':
        adj_matrix, dis_matrix = get_adj_dis_matrix(DATASET, ADJ_FILE, args.num_nodes, False, "./dataset/PEMS03/PEMS03.txt")
        if args.adj_norm:
            adj_matrix, dis_matrix = norm_adj(adj_matrix), norm_adj(dis_matrix)
    elif args.dataset.lower() in ['metr-la', 'pems-bay']:
        adj_matrix, dis_matrix = None, None
    else:
        adj_matrix, dis_matrix = get_adj_dis_matrix(DATASET, ADJ_FILE, args.num_nodes, False)
        if args.adj_norm:
            adj_matrix, dis_matrix = norm_adj(adj_matrix), norm_adj(dis_matrix)
    if adj_matrix is not None and dis_matrix is not None:
        # compute spatial-temporal adjacent and distance matrix
        st_adj = construct_adj(adj_matrix,2)
        st_dis = construct_adj(dis_matrix,2)
        st_adj = TensorFloat(st_adj).to(args.device)
        st_dis = TensorFloat(st_dis).to(args.device)
        adj_matrix, dis_matrix = TensorFloat(adj_matrix),TensorFloat(dis_matrix)

    # init generator and discriminator model
    generator = Network(st_adj,st_dis,args.num_nodes,args.input_dim,args.output_dim,args.in_steps,args.out_steps,
                        args.embed_dim,args.rnn_units,args.num_layers,args.input_embedding_dim,
                        args.periods_embedding_dim,args.weekend_embedding_dim,args.holiday_embedding_dim,
                        args.spatial_embedding_dim,args.adaptive_embedding_dim,args.dim_embed_feature,args.periods,
                        args.predict_time, args.gat_hidden, args.mlp_hidden, args.gat_drop, args.gat_heads,
                        args.gat_alpha, args.gat_concat, args.mlp_act, args.mlp_drop,args.num_gat,args.num_back)
    generator = generator.to(args.device)
    generator = init_model(generator)

    discriminator = Discriminator(args)
    discriminator = discriminator.to(args.device)
    discriminator = init_model(discriminator)
    #
    discriminator_spatial = Discriminator_spatial(args)
    discriminator_spatial = discriminator_spatial.to(args.device)
    discriminator_spatial = init_model(discriminator_spatial)

    discriminator_temporal = Discriminator_temporal(args)
    discriminator_temporal = discriminator_temporal.to(args.device)
    discriminator_temporal = init_model(discriminator_temporal)
    if args.dataset in ['METR-LA', 'PEMS-Bay']:
        train_loader, val_loader, test_loader, scaler = get_dataloader_meta_la(args,
                                                                    normalizer=args.normalizer,
                                                                    tod=args.tod,
                                                                    dow=False,
                                                                    weather=False,
                                                                    single=False)
    # load dataset X = [B', W, N, D], Y = [B', H, N, D]
    else:
        train_loader, val_loader, test_loader, scaler = get_dataloader_pems(DATASET,args.batch_size,
                            args.val_ratio,args.test_ratio,args.in_steps,args.out_steps,
                            args.flow_dim,args.period_dim,args.weekend_dim,args.holiday_dim,
                            args.hop_dim,args.weather_dim)

    # loss function
    if args.loss_func == 'mask_mae':
        loss_G = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss_G = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss_G = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'huber':
        loss_G = torch.nn.HuberLoss().to(args.device)
    else:
        raise ValueError
    loss_D = torch.nn.BCELoss()

    # optimizer
    optimizer_G = torch.optim.Adam(params=generator.parameters(),
                                   lr=args.lr_init,
                                   eps=1.0e-8,
                                   weight_decay=0,
                                   amsgrad=False)

    optimizer_D = torch.optim.Adam(params=discriminator.parameters(),
                                   lr=args.lr_init*0.1,
                                   eps=1.0e-8,
                                   weight_decay=0,
                                   amsgrad=False)
    #
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

    # learning rate decay scheduler
    lr_scheduler_G, lr_scheduler_D, lr_scheduler_spatial,lr_scheduler_temporal = None, None, None,None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_G,
                                                              milestones=lr_decay_steps,
                                                              gamma=args.lr_decay_rate)

        lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D,
                                                              milestones=lr_decay_steps,
                                                              gamma=args.lr_decay_rate)

        lr_scheduler_spatial = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_spatial,
                                                                 milestones=lr_decay_steps,
                                                                 gamma=args.lr_decay_rate)
        lr_scheduler_temporal = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_temporal,
                                                                 milestones=lr_decay_steps,
                                                                 gamma=args.lr_decay_rate)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

    # config log path
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir, 'exps/logs', args.dataset, current_time)
    args.log_dir = log_dir

    # model training or testing
    trainer = Trainer(args,
                      generator, discriminator, discriminator_spatial,discriminator_temporal,
                      train_loader, val_loader, test_loader, scaler,
                      loss_G, loss_D,
                      optimizer_G, optimizer_D, optimizer_spatial,optimizer_temporal,
                      lr_scheduler_G, lr_scheduler_D, lr_scheduler_spatial,lr_scheduler_temporal)

    if args.mode.lower() == 'train':
        trainer.train()
    elif args.mode.lower() == 'test':
        # generator.load_state_dict(torch.load('./log/{}/20221128054144/best_model.pth'.format(args.dataset)))
        print("Load saved model")
        trainer.test(generator, trainer.args, test_loader, scaler, trainer.logger, path=f'./exps/log/{args.dataset}/20221130052054/')
    else:
        raise ValueError
