import argparse
import configparser

import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
            self,
            num_nodes,
            batch_size,
            periods_embedding_dim,
            weekend_embedding_dim,
            input_dim,                      # flow, day, weekend, holiday
            periods=288,
            weekend=7,
            embed_dim=12,
            in_steps=12,
    ):
        super(Encoder, self).__init__()

        self.num_nodes = num_nodes
        self.periods = periods
        self.weekend = weekend
        # 周期的embedding维度
        self.periods_embedding_dim = periods_embedding_dim
        # 每周的embedding维度
        self.weekend_embedding_dim = weekend_embedding_dim
        self.in_steps = in_steps
        self.input_dim = input_dim
        # period的embedding
        if periods_embedding_dim>0:
            self.periods_embedding = nn.Embedding(periods, periods_embedding_dim)
        # 每周的embedding
        if weekend_embedding_dim>0:
            self.weekend_embedding = nn.Embedding(weekend, weekend_embedding_dim)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.time_embeddings = nn.Parameter(torch.randn(batch_size,in_steps, embed_dim), requires_grad=True)
    def forward(self, x):
        '''
        获取当前的动态图
        :param x:
        shape:b,ti,n,di
        :return:
        shape:b,to,n,do
        '''
        batch_size = x.shape[0]
        node_embedding = self.node_embeddings
        time_embedding = self.time_embeddings[:batch_size]

        if self.periods_embedding_dim > 0:
            periods = x[..., 1]
        if self.weekend_embedding_dim > 0:
            weekend = x[..., 2]
        if self.periods_embedding_dim > 0:
            periods_emb = self.periods_embedding(
                (periods*self.periods).long()
            )
            time_embedding = torch.mul(time_embedding, periods_emb[:,:,0])
        if self.weekend_embedding_dim > 0:
            weekend_emb = self.weekend_embedding(
                weekend.long()
            )  # (batch_size, in_steps, num_nodes, weekend_embedding_dim)
            time_embedding = torch.mul(time_embedding, weekend_emb[:,:,0])
        embeddings = [node_embedding,time_embedding]
        return embeddings

class MGSTGNN(nn.Module):
    def __init__(
            self,
            num_nodes,              #节点数
            batch_size,
            input_dim,              #输入维度
            rnn_units,              #GRU循环单元数
            output_dim,             #输出维度
            num_grus,               #GRU的层数
            embed_dim,              #GNN嵌入维度
            in_steps=12,            #输入的时间长度
            out_steps=12,           #预测的时间长度
            predict_time=1,
            use_back=False,
            periods=288,
            weekend=7,
            periods_embedding_dim=12,
            weekend_embedding_dim=12,
            num_input_dim=1,
    ):
        super(MGSTGNN, self).__init__()
        assert num_input_dim <= input_dim
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.num_input_dim = num_input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.embed_dim = embed_dim
        self.num_grus = num_grus
        self.encoder = Encoder(num_nodes, batch_size, periods_embedding_dim, weekend_embedding_dim, input_dim, periods,
                               weekend, embed_dim, in_steps)

        self.predictor = DSTRNN(num_nodes, num_input_dim, rnn_units, embed_dim, num_grus, in_steps, dim_out=output_dim,
                                use_back=use_back,conv_steps=predict_time)

        self.predict_time = predict_time
    def forward(self, source):
        batch_size = source.shape[0]
        embeddings = self.encoder(source)
        init_state = self.predictor.init_hidden(batch_size)#,self.num_node,self.hidden_dim
        _, output = self.predictor(source[...,:self.num_input_dim], init_state, embeddings) # B, T, N, hidden

        return output

class DSTRNN(nn.Module):
    def __init__(self, node_num, dim_in, hidden_dim, embed_dim, num_grus, in_steps=12, dim_out=1,
            use_back=False, conv_steps=2, conv_bias=True):
        super(DSTRNN, self).__init__()
        assert len(num_grus) >= 1, 'At least one GRU layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.hidden_dim = hidden_dim
        self.use_back = use_back
        self.num_grus = num_grus
        self.grus = nn.ModuleList([
            GRUCell(node_num, dim_in, hidden_dim, embed_dim)
            for _ in range(sum(num_grus))
        ])
        if use_back>0:
            self.backs = nn.ModuleList([
                nn.Linear(hidden_dim,dim_in)
                for _ in range(sum(num_grus))
            ])
        # predict output
        self.predictors = nn.ModuleList([
             # nn.Linear(hidden_dim,dim_cat)
             nn.Conv2d(conv_steps, dim_out * in_steps, kernel_size=(1,hidden_dim), bias=conv_bias)
             for _ in num_grus
        ])
        # skip
        self.skips = nn.ModuleList([
            # nn.Linear(hidden_dim,dim_in)
            nn.Conv2d(conv_steps, dim_in * in_steps, kernel_size=(1,hidden_dim), bias=conv_bias)
            for _ in range(len(num_grus)-1)
        ])
        # dropout
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=0.1)
            for _ in range(len(num_grus))
        ])
        self.conv_steps = conv_steps
    def forward(self, x, init_state, embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (len(num_grus), B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim

        outputs = []
        skip = x
        seq_length = x.shape[1] # T
        current_inputs = x

        index1 = 0
        init_hidden_states = [state.to(x.device) for state in init_state]
        batch_size,time_stamps,node_num,dimensions = skip.size()


        for i in range(len(self.num_grus)):
            inner_states = []
            for t in range(seq_length):
                index2 = t % self.num_grus[i]
                prev_state = init_hidden_states[index1+index2]
                inp = current_inputs[:, t, :, :]
                if self.use_back:
                    inp = inp - self.backs[index1+index2](prev_state)
                init_hidden_states[index1+index2] = self.grus[index1+index2](inp, prev_state, [embeddings[0], embeddings[1][:, t, :]]) # [B, N, hidden_dim]
                inner_states.append(init_hidden_states[index1+index2])
            index1 += self.num_grus[i]
            current_inputs = torch.stack(inner_states, dim=1) # [B, T, N, D]
            current_inputs = self.dropouts[i](current_inputs[:, -self.conv_steps:, :, :])
            outputs.append(self.predictors[i](current_inputs).reshape(batch_size,time_stamps,node_num,-1))
            if i < len(self.num_grus)-1:
                current_inputs = skip - self.skips[i](current_inputs).reshape(batch_size,time_stamps,node_num,-1)

        predict = outputs[0]
        for i in range(1,len(outputs)):
            predict = predict + outputs[i]
        return None, predict

    def init_hidden(self, batch_size):
        init_states = []
        index = 0
        for i in range(len(self.num_grus)):
            for j in range(self.num_grus[i]):
                init_states.append(self.grus[index+j].init_hidden_state(batch_size))
            index += self.num_grus[i]
        return init_states # [sum(num_grus), B, N, hidden_dim]

class GRUCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, embed_dim):
        super(GRUCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GCN(dim_in+self.hidden_dim, dim_out*2, embed_dim, node_num)
        self.update = GCN(dim_in+self.hidden_dim, dim_out, embed_dim, node_num)
    def forward(self, x, state, embeddings):

        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        input_and_state = torch.cat((x, state), dim=-1) # [B, N, 1+D]
        z_r = torch.sigmoid(self.gate(input_and_state, embeddings))
        z,r = torch.split(z_r,self.hidden_dim,dim=-1)
        # r = torch.sigmoid(self.gate_r(input_and_state, node_embeddings,time_embeddings))
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, embeddings))
        h = r*state + (1-r)*hc


        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim, node_num):
        super(GCN, self).__init__()
        self.node_num = node_num
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, 2, dim_in, dim_out)) # [D, C, F]
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out)) # [D, F]
        self.norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.drop = nn.Dropout(0.1)
    def forward(self, x, embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D], time_embeddings shaped [b,d]
        # output shape [B, N, C]
        node_embeddings, time_embeddings = embeddings[0],embeddings[1]
        supports1 = torch.eye(self.node_num).to(x.device)
        embedding = self.drop(
            self.norm(node_embeddings.unsqueeze(0) + time_embeddings.unsqueeze(1)))  # torch.mul(node_embeddings, node_time)
        supports2 = F.softmax(torch.matmul(embedding, embedding.transpose(1, 2)), dim=2)    # b,n,n
        x_g1 = torch.einsum("nm,bmc->bnc", supports1, x)
        x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)
        x_g = torch.stack([x_g1,x_g2],dim=1)        # b,2,n,d
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) # N, dim_in, dim_out
        bias = torch.einsum('bd,do->bo', time_embeddings, self.bias_pool) # b, N, dim_out
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias.unsqueeze(1)  #b, N, dim_out

        return x_gconv

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.mgstgnn = MGSTGNN(args.num_nodes,args.batch_size,args.input_dim,args.rnn_units,args.output_dim,args.num_grus,args.embed_dim,
            in_steps=args.in_steps,out_steps=args.out_steps,predict_time=args.predict_time,use_back=args.use_back,
            periods=args.periods,weekend=args.weekend,periods_embedding_dim=args.periods_embedding_dim,
            weekend_embedding_dim=args.weekend_embedding_dim,num_input_dim=args.num_input_dim)
    def forward(self,x):

        out = self.mgstgnn(x)
        return out

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('--dataset', default='PEMS03', type=str)
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

    args.add_argument('--num_input_dim', default=config['model']['num_input_dim'], type=int)

    args.add_argument('--periods_embedding_dim', default=config['model']['periods_embedding_dim'], type=int)
    args.add_argument('--weekend_embedding_dim', default=config['model']['weekend_embedding_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    args.add_argument('--num_grus', default=config['model']['num_grus'], type=str)
    args.add_argument('--periods', default=config['model']['periods'], type=int)
    args.add_argument('--weekend', default=config['model']['weekend'], type=int)
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
    from utils.util import init_seed
    init_seed(args.seed)
    args.num_grus = [int(i) for i in list(args.num_grus.split(','))]
    network = Network(args)
    # network = DDGCRN(307,1,64,1,2,10,12,12,1)
    summary(network, [args.batch_size, args.in_steps, args.num_nodes, args.input_dim])
