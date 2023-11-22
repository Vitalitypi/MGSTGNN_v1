import argparse
import configparser
import math
import time
import torch
import yaml
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import MultiheadAttention

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out

class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class Network(nn.Module):
    def __init__(self, args,
                 feed_forward_dim=256,
                 num_heads=4,
                 dropout=0.1,
                 num_layers=3
                 ):
        super(Network, self).__init__()
        # period的embedding
        self.period_embedding = nn.Embedding(288, 32)
        # 每周的embedding
        self.weekend_embedding = nn.Embedding(7, 24)
        # 节日的embedding
        self.holiday_embedding = nn.Embedding(2, 16)

        # 输入数据的映射
        self.flow_proj = nn.Linear(1, 24)
        # 输入数据的映射
        self.hops_proj = nn.Linear(1, 16)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(args.in_steps, args.num_nodes, 32))
        )
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(72, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(72, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers_t2 = nn.ModuleList(
            [
                SelfAttentionLayer(144, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers_s2 = nn.ModuleList(
            [
                SelfAttentionLayer(144, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(
            args.in_steps *(144), args.out_steps * args.output_dim
        )
    def forward(self,x):
        batch_size = x.size(0)
        period_emb = self.period_embedding((x[..., 1]*288).long())
        weekend_emb = self.weekend_embedding(x[..., 2].long())
        holiday_emb = self.holiday_embedding(x[..., 3].long())
        time_emb = torch.cat([period_emb,weekend_emb,holiday_emb],dim=-1)
        flow_emb = self.flow_proj(x[...,:1])
        hops_emb = self.hops_proj(x[...,4:])
        adp_emb = self.adaptive_embedding.expand(
            size=(batch_size, *self.adaptive_embedding.shape)
        )
        space_emb = torch.cat([flow_emb,hops_emb,adp_emb],dim=-1)
        res_t,res_s = time_emb,space_emb
        for attn in self.attn_layers_t:
            time_emb = attn(time_emb, dim=1)
        for attn in self.attn_layers_s:
            space_emb = attn(space_emb, dim=2)
        out = torch.tan(time_emb)*torch.sigmoid(space_emb)
        res_t = res_t + out
        res_s = res_s + out
        out = torch.cat([res_t,res_s],dim=-1)
        batch_size,time_stamp,num_nodes,dimensions = out.size()
        for attn in self.attn_layers_t2:
            out = attn(out, dim=1)
        for attn in self.attn_layers_s2:
            out = attn(out, dim=2)
        out = out.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
        out = out.reshape(
            batch_size, num_nodes, time_stamp * dimensions
        )
        out = self.output_proj(out).view(
            batch_size, num_nodes, time_stamp, -1
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        return out
if __name__ == "__main__":
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('--dataset', default='PEMS08', type=str)
    args.add_argument('--mode', default='train', type=str)
    args.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
    args.add_argument('--debug', default='False', type=eval)
    args.add_argument('--model', default='MGSTGNN', type=str)
    args.add_argument('--cuda', default=True, type=bool)
    args1 = args.parse_args()

    #get configuration
    config_file = '../config/{}.conf'.format(args1.dataset)
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

    network = Network(args)
    # network = DDGCRN(307,1,64,1,2,10,12,12,1)
    summary(network, [args.batch_size, args.in_steps, args.num_nodes, args.input_dim])
