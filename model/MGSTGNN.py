import argparse
import configparser
import time
import torch
import yaml
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from collections import OrderedDict
class Encoder(nn.Module):
    def __init__(
            self,
            num_nodes,
            input_embedding_dim,
            periods_embedding_dim,
            weekend_embedding_dim,
            holiday_embedding_dim,
            spatial_embedding_dim,
            adaptive_embedding_dim,
            dim_embed_feature,              # embedding dimension of features
            input_dim,                      # flow, day, weekend, holiday

            periods,
            in_steps=12,
            use_mixed_proj=True,
    ):
        super(Encoder, self).__init__()

        assert dim_embed_feature == input_embedding_dim+sum(periods_embedding_dim)+weekend_embedding_dim + \
               holiday_embedding_dim+spatial_embedding_dim+adaptive_embedding_dim , \
            'The total dimension is not equal to the sum of the each dimension! '

        self.num_nodes = num_nodes
        self.num_periods = len(periods)
        self.periods = periods
        # 输入的embedding维度
        self.input_embedding_dim = input_embedding_dim
        # 周期的embedding维度
        self.periods_embedding_dim = periods_embedding_dim
        # 每周的embedding维度
        self.weekend_embedding_dim = weekend_embedding_dim
        self.holiday_embedding_dim = holiday_embedding_dim
        # 空间的embedding维度
        self.spatial_embedding_dim = spatial_embedding_dim
        # 自适应的embedding维度
        self.adaptive_embedding_dim = adaptive_embedding_dim
        # 编码后的总维度
        self.model_dim = (
                input_embedding_dim
                + sum(periods_embedding_dim)
                + weekend_embedding_dim
                + holiday_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.in_steps = in_steps
        self.input_dim = input_dim
        # 输入数据的映射
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        # period的embedding
        self.periods_embedding = nn.ModuleList([
            nn.Embedding(periods[i], periods_embedding_dim[i]) for i in range(self.num_periods)
        ])
        # 每周的embedding
        self.weekend_embedding = nn.Embedding(7, weekend_embedding_dim)
        # 节日的embedding
        self.holiday_embedding = nn.Embedding(2, holiday_embedding_dim)
        # 节点的embedding
        self.node_emb = nn.Parameter(
            torch.empty(num_nodes, self.spatial_embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_emb)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(in_steps, num_nodes, self.adaptive_embedding_dim))
        )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                12 * self.model_dim, 12 * 1
            )
        else:
            self.temporal_proj = nn.Linear(12, 12)
            self.output_proj = nn.Linear(self.model_dim, 1)
        self.skip = nn.Linear(input_dim,dim_embed_feature)
        self.ln = nn.LayerNorm(dim_embed_feature)
    def forward(self,x):
        '''
        将输入x映射到高维空间
        :param x:
        shape:b,ti,n,di
        :return:
        shape:b,to,n,do
        '''
        batch_size = x.shape[0]
        periods = []
        index = 1
        for i in range(self.num_periods):
            periods.append(x[..., index])
            index+=1
        if self.weekend_embedding_dim > 0:
            weekend = x[..., index]
            index+=1
        if self.holiday_embedding_dim > 0:
            holiday = x[..., index]
            index+=1
        x = x[..., : self.input_dim]
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        for i in range(self.num_periods):
            period = periods[i]
            period_emb = self.periods_embedding[i](
                (period*self.periods[i]).long()
            )
            features.append(period_emb)
        if self.weekend_embedding_dim > 0:
            weekend_emb = self.weekend_embedding(
                weekend.long()
            )  # (batch_size, in_steps, num_nodes, weekend_embedding_dim)
            features.append(weekend_emb)
        if self.holiday_embedding_dim > 0:
            holiday_emb = self.holiday_embedding(
                holiday.long()
            )
            features.append(holiday_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        return x

class DDGCRN(nn.Module):
    def __init__(
            self,
            num_nodes,              #节点数
            input_dim,              #输入维度
            rnn_units,              #GRU循环单元数
            output_dim,             #输出维度
            num_grus,               #GRU的层数
            embed_dim,              #GNN嵌入维度
            in_steps=12,            #输入的时间长度
            out_steps=12,           #预测的时间长度
            predict_time=1,
            use_back=False,
            use_D=True,
            use_W=True,
            use_H=False,
    ):
        super(DDGCRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.embed_dim = embed_dim
        self.num_grus=num_grus
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)
        if use_D:
            self.T_i_D_emb = nn.Parameter(torch.empty(288, embed_dim))
        if use_W:
            self.D_i_W_emb = nn.Parameter(torch.empty(7, embed_dim))
        if use_H:
            self.Holiday_emb = nn.Parameter(torch.empty(2, embed_dim))
        self.use_D,self.use_W,self.use_H = use_D,use_W,use_H
        self.encoder = DSTRNN(num_nodes, 1, rnn_units, embed_dim,num_grus, 12, use_back=use_back,
                              conv_steps=predict_time)

        #self.norm = nn.LayerNorm(args.rnn_units*len(self.num_grus), eps=1e-12)
        #self.out_dropout = nn.Dropout(0.1)

        #self.end_conv = nn.Conv2d(predict_time, 12 * self.output_dim, kernel_size=(1, args.rnn_units*len(self.num_grus)), bias=True)

        self.predict_time = predict_time
    def forward(self, source, A=None):
        node_embedding = self.node_embeddings
        if self.use_D:
            t_i_d_data   = source[..., 1]
            T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]
            node_embedding = torch.mul(node_embedding, T_i_D_emb)
        if self.use_W:
            d_i_w_data   = source[..., 2]
            D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]
            node_embedding = torch.mul(node_embedding, D_i_W_emb)
        if self.use_H:
            holiday_data = source[..., 3]
            Holiday_emb = self.Holiday_emb[(holiday_data).type(torch.LongTensor)]
            node_embedding = torch.mul(node_embedding, Holiday_emb)
        node_embeddings=[node_embedding,self.node_embeddings]
        # b,t,n,d = source.size()
        # source: B, T, N, D
        init_state = self.encoder.init_hidden(source.shape[0])#,self.num_node,self.hidden_dim
        source = source[..., 0].unsqueeze(-1)
        # source = torch.stack([source[..., 0],source[..., 1]], dim=-1)
        _, output = self.encoder(source, init_state, node_embeddings) # B, T, N, hidden

        #output = self.out_dropout(self.norm(output[:, -self.predict_time:, :, :])) # B, r, N, hidden

        # CNN based predictor
        #output = self.end_conv((output)) # B, T*C, N, d'
        #output = output.squeeze(-1).reshape(-1, self.out_steps, self.output_dim, self.num_node)
        #output = output.permute(0, 1, 3, 2) # B, T, N, C


        return output

class DSTRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, embed_dim, num_grus, in_steps=12,
                 use_back=False, conv_steps=2, conv_bias=True):
        super(DSTRNN, self).__init__()
        assert len(num_grus) >= 1, 'At least one GRU layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.dim_out = dim_out
        self.use_back = use_back
        self.num_grus = num_grus
        self.grus = nn.ModuleList([
            GRUCell(node_num, dim_in, dim_out, embed_dim)
            for _ in range(sum(num_grus))
        ])
        if use_back>0:
            self.backs = nn.ModuleList([
                nn.Linear(dim_out,dim_out)
                for _ in range(sum(num_grus))
            ])
        # predict output
        self.predictors = nn.ModuleList([
             # nn.Linear(dim_out,dim_cat)
             nn.Conv2d(conv_steps, 1 * in_steps, kernel_size=(1,dim_out), bias=conv_bias)
             for _ in num_grus
        ])
        # skip
        self.skips = nn.ModuleList([
            # nn.Linear(dim_out,dim_in)
            nn.Conv2d(conv_steps, dim_in * in_steps, kernel_size=(1,dim_out), bias=conv_bias)
            for _ in range(len(num_grus)-1)
        ])
        # norms
        #self.norms = nn.ModuleList([
          #  nn.LayerNorm(dim_in)
          #  for _ in range(len(num_grus)-1)
        #])
        # dropout
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=0.1)
            for _ in range(len(num_grus))
        ])
        self.conv_steps = conv_steps
    def forward(self, x, init_state, node_embeddings):
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
                init_hidden_states[index1+index2] = self.grus[index1+index2](inp, prev_state, [node_embeddings[0][:, t, :, :], node_embeddings[1]]) # [B, N, hidden_dim]
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
    def forward(self, x, state, node_embeddings):

        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        input_and_state = torch.cat((x, state), dim=-1) # [B, N, 1+D]
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z,r = torch.split(z_r,self.hidden_dim,dim=-1)
        # r = torch.sigmoid(self.gate_r(input_and_state, node_embeddings,time_embeddings))
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
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
        self.dim_hidden1 = 16
        self.dim_hidden2 = 2
        self.embed_dim = embed_dim
        self.fc=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(dim_in, self.dim_hidden1)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.dim_hidden1, self.dim_hidden2)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.dim_hidden2, self.embed_dim))]))

    def forward(self, x, node_embeddings):


        # x shaped[B, N, C], node_embeddings shaped [[b, N, D], [N, D]]
        # output shape [B, N, C]
        supports1 = torch.eye(self.node_num).to(x.device)
        x_ = self.fc(x)
        nodevec = torch.tanh(torch.mul(node_embeddings[0], x_))  #[B,N,dim_in]
        supports2 = GCN.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))), supports1)
        x_g1 = torch.einsum("nm,bmc->bnc", supports1, x)
        x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)
        x_g = torch.stack([x_g1,x_g2],dim=1)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
                                                                                  #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
        bias = torch.matmul(node_embeddings[1], self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
        # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]
        return x_gconv

    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        # encoder
        # self.encoder = Encoder(num_nodes,input_embedding_dim,periods_embedding_dim,weekend_embedding_dim,
        #                        holiday_embedding_dim,spatial_embedding_dim,adaptive_embedding_dim,
        #                        dim_embed_feature,input_dim,periods)
        self.mgstgnn = DDGCRN(args.num_nodes,args.input_dim,args.rnn_units,args.output_dim,args.num_grus,args.embed_dim,
                              in_steps=args.in_steps,out_steps=args.out_steps,predict_time=args.predict_time,use_back=args.use_back)
    def forward(self,x):

        # 进行encoding
        # enc = self.encoder(x)
        # dat = torch.cat((x,enc),dim=-1)
        out = self.mgstgnn(x)
        return out

if __name__ == "__main__":
    Mode = 'Train'
    DATASET = 'PEMS03'
    MODEL = "MGSTGNN"
    DEBUG = 'True'
    # get configuration
    config_file = '../config/{}.yaml'.format(DATASET)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
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
    parser.add_argument('--use_day', default=config['data']['use_day'], type=eval)
    parser.add_argument('--use_week', default=config['data']['use_week'], type=eval)
    parser.add_argument('--use_holiday', default=config['data']['use_holiday'], type=eval)

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
    parser.add_argument('--use_discriminator', default=config['model']['use_discriminator'], type=eval)

    # parser.add_argument('--dim_embed_feature', default=config['model']['dim_embed_feature'], type=int)
    # parser.add_argument('--input_embedding_dim', default=config['model']['input_embedding_dim'], type=int)
    # parser.add_argument('--periods_embedding_dim', default=config['model']['periods_embedding_dim'], type=list)
    # parser.add_argument('--weekend_embedding_dim', default=config['model']['weekend_embedding_dim'], type=int)
    # parser.add_argument('--holiday_embedding_dim', default=config['model']['holiday_embedding_dim'], type=int)
    # parser.add_argument('--spatial_embedding_dim', default=config['model']['spatial_embedding_dim'], type=int)
    # parser.add_argument('--adaptive_embedding_dim', default=config['model']['adaptive_embedding_dim'], type=int)

    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_grus', default=config['model']['num_grus'], type=list)
    parser.add_argument('--periods', default=config['model']['periods'][:config['model']['period_dim']], type=list)
    parser.add_argument('--predict_time', default=config['model']['predict_time'], type=int)
    parser.add_argument('--use_back', default=config['model']['use_back'], type=eval)
    # train
    parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    parser.add_argument('--random', default=config['train']['random'], type=eval)
    parser.add_argument('--seed', default=config['train']['seed'], type=int)
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
    network = Network(args)
    # network = DDGCRN(307,1,64,1,2,10,12,12,1)
    summary(network, [args.batch_size, args.in_steps, args.num_nodes, args.input_dim])
