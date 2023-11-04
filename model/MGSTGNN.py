import math
import numpy as np
import torch
from torch import nn
from torchinfo import summary
from utils.dataloader import get_adj_dis_matrix
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=4,                        # flow, day, week, holiday
            output_dim=1,
            input_embedding_dim=80,
            tod_embedding_dim=24,
            dow_embedding_dim=8,
            spatial_embedding_dim=0,
            use_mixed_proj=True,
            dim_embed_feature=80,              # embedding dimension of features
    ):
        super(Encoder, self).__init__()
        # encoder
        self.steps_per_day = steps_per_day
        # 输入的embedding维度
        self.input_embedding_dim = input_embedding_dim
        # 每天的embedding维度
        self.tod_embedding_dim = tod_embedding_dim
        # 每周的embedding维度
        self.dow_embedding_dim = dow_embedding_dim
        # 空间的embedding维度
        self.spatial_embedding_dim = spatial_embedding_dim
        # 自适应的embedding维度
        self.adaptive_embedding_dim = dim_embed_feature-input_embedding_dim-tod_embedding_dim-dow_embedding_dim-spatial_embedding_dim
        # 编码后的总维度
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + self.adaptive_embedding_dim
        )
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 输入数据的映射
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        # 每天的embedding
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        # 每周的embedding
        self.dow_embedding = nn.Embedding(2, (dow_embedding_dim * 3) // 4)
        # 节日的embedding
        self.hol_embedding = nn.Embedding(2, dow_embedding_dim // 4)
        # 节点的embedding
        self.node_emb = nn.Parameter(
            torch.empty(num_nodes, self.spatial_embedding_dim)
        )
        self.num_nodes = num_nodes
        self.use_mixed_proj = use_mixed_proj
        nn.init.xavier_uniform_(self.node_emb)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(in_steps, num_nodes, self.adaptive_embedding_dim))
        )
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)
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
        residual = x
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
            hol = x[..., 3]
        x = x[..., : self.input_dim]
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
            hol_emb = self.hol_embedding(
                hol.long()
            )
            features.append(hol_emb)
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

class DSTGNN(nn.Module):
    def __init__(
            self,
            num_nodes,              #节点数
            input_dim,              #输入维度
            rnn_units,              #GRU循环单元数
            output_dim,             #输出维度
            in_steps,               #输入的时间长度
            out_steps,              #预测的时间长度
            num_layers,             #GRU的层数
            embed_dim,              #GNN嵌入维度
            recent_stamp=2,
    ):
        super(DSTGNN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)
        self.time_embeddings = nn.Parameter(torch.randn(self.in_steps, embed_dim), requires_grad=True)

        self.encoder = DSTRNN(num_nodes, input_dim, rnn_units, embed_dim, num_layers)

        self.layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(0.1)

        self.end_conv = nn.Conv2d(recent_stamp, out_steps * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.recent_stamp = recent_stamp
    def forward(self, source):
        b,t,n,d = source.size()
        # source: B, T, N, D
        init_state = self.encoder.init_hidden(source.shape[0])#,self.num_node,self.hidden_dim
        output, _ = self.encoder(source, init_state, self.node_embeddings, self.time_embeddings) # B, T, N, hidden
        output = self.out_dropout(self.layernorm(output[:, -self.recent_stamp:, :, :])) # B, r, N, hidden

        # CNN based predictor
        output = self.end_conv((output)) # B, T*C, N, d'
        output = output.squeeze(-1).reshape(-1, self.out_steps, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2) # B, T, N, C


        return output

class DSTRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, embed_dim, num_layers=1, num_gru=2, num_norms=12):
        super(DSTRNN, self).__init__()
        assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dim_out = dim_out
        self.gru0 = GRUCell(node_num, dim_in, dim_out, embed_dim)
        self.grus = nn.ModuleList([
            GRUCell(node_num, dim_out, dim_out, embed_dim)
            for _ in range(num_gru)
        ])
        self.num_gru = num_gru
        self.gats = nn.ModuleList([GAT(dim_in,dim_out) for _ in range(num_norms)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim_out) for _ in range(num_norms)])
    def forward(self, x, init_state, node_embeddings, time_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1] # T
        current_inputs = x
        output_hidden = []
        state = init_state[0]
        inner_states = []
        prev = x[:,0]
        for t in range(seq_length):
            state = self.gru0(current_inputs[:, t, :, :], state, node_embeddings, time_embeddings[t]) # [B, N, hidden_dim]
            # inner_states.append(state)
            att = self.gats[t](current_inputs[:, t, :, :], prev)
            inner_states.append(self.norms[t](state+att))

            prev = x[:,t]
        base_state = state
        current_inputs = torch.stack(inner_states, dim=1) # [B, T, N, D]
        states = [init_state[1] for _ in range(self.num_gru)]
        for t in range(seq_length):
            gru = self.grus[t%self.num_gru]
            prev_state = states[t%self.num_gru]
            states[t%self.num_gru] = gru(current_inputs[:, t, :, :], prev_state, node_embeddings, time_embeddings[t]) # [B, N, hidden_dim]
        states.append(base_state)
        current_inputs = torch.stack(states, dim=1) # [B, num_gru+1, N, D]
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.gru0.init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0) # (num_layers, B, N, hidden_dim)

class GRUCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, embed_dim):
        super(GRUCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate_r = GCN(dim_in+self.hidden_dim, dim_out, embed_dim, node_num)
        self.gate_z = GCN(dim_in+self.hidden_dim, dim_out, embed_dim, node_num)
        self.update = GCN(dim_in+self.hidden_dim, dim_out, embed_dim, node_num)

    def forward(self, x, state, node_embeddings, time_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1) # [B, N, 1+D]
        z = torch.sigmoid(self.gate_z(input_and_state, node_embeddings,time_embeddings))
        r = torch.sigmoid(self.gate_r(input_and_state, node_embeddings,time_embeddings))
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, time_embeddings))
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
        # self.adj,dis = get_adj_dis_matrix('./dataset/PEMS04/PEMS04.csv', 307, False)
        # self.dis = dis/np.max(dis)
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.embs_dropout = nn.Dropout(0.1)
    def forward(self, x, node_embeddings, time_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D], embedding shaped [N, N]
        # output shape [B, N, C]
        node_embeddings = self.embs_dropout(
            self.layernorm(node_embeddings + time_embeddings.unsqueeze(0)))  # torch.mul(node_embeddings, node_time)
        embedding = F.softmax(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)), dim=1)
        # emb = embedding+torch.eye(self.node_num).to(x.device)
        # adj = torch.tensor(self.adj).to(x.device)
        # dis = torch.tensor(self.dis).to(x.device).to(torch.float)
        support_set = [torch.eye(self.node_num).to(x.device), embedding]
        supports = torch.stack(support_set, dim=0)  # [3, N, N]
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) # N, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool) # N, dim_out

        x_g = torch.einsum("knm,bmc->bknc", supports, x) # B, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias # B, N, dim_out
        return x_gconv

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, num_nodes=307, concat=True, dataset='PEMS04'):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_nodes = num_nodes
        adj,dis = get_adj_dis_matrix("PEMS04".format(dataset,dataset),num_nodes)
        I = torch.eye(self.num_nodes)
        adj = torch.Tensor(adj)
        adj = F.pad(
                adj,
                (self.num_nodes,0,self.num_nodes,0),
                "constant", 0,
            )
        # adj[self.num_nodes:,:self.num_nodes] = I
        adj[:self.num_nodes,self.num_nodes:] = I
        adj[:self.num_nodes,:self.num_nodes] = adj[self.num_nodes:,self.num_nodes:]

        self.adj = adj
        dis = torch.Tensor(dis/np.max(dis))
        dis = F.pad(
                dis,
                (self.num_nodes,0,self.num_nodes,0),
                "constant", 0,
            )
        dis[:self.num_nodes,self.num_nodes:] = I
        dis[:self.num_nodes,:self.num_nodes] = dis[self.num_nodes:,self.num_nodes:]
        self.dis = dis
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, ht):
        '''
        Args:
            h: b,n,d
            ht:b,n,d
        Returns:
        '''
        h = torch.cat([ht,h],dim=1)         # b,2n,d
        adj = self.adj.to(h.device)
        dis = self.dis.to(h.device)
        w  = self.W.to(h.device)
        Wh = torch.einsum("bni,io->bno",h,w)                    # h.shape: (b, 2N, in_features), Wh.shape: (b, 2N, out_features)

        # Wh = torch.mm(h, w)
        e = self._prepare_attentional_mechanism_input(Wh)       # b,2n,2n

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)           # b,2n,2n
        # 乘以权重矩阵
        # attention = attention * dis
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        h_prime = torch.cat([h_prime[:,:self.num_nodes],h_prime[:,self.num_nodes:]],dim=-1)
        # h_prime = h_prime[:,self.num_nodes:]
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (b, 2N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (b, 2N, 1)
        # e.shape (b, 2N, 2N)
        a = self.a.to(Wh.device)
        Wh1 = torch.matmul(Wh, a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0,2,1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.abs(x)
        x = self.drop(x)
        return x

class GAT(nn.Module):
    def __init__(
            self, dim_in, dim_out, dim_hidden=256, dropout=0.6, heads=1, alpha=0.2,bias=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(dim_in, dim_hidden, dropout=dropout, alpha=alpha, concat=True) for _ in range(heads)]
        self.output = Mlp(dim_hidden*heads*2,dim_hidden,dim_out)
        # self.output = nn.Conv2d(dim_hidden*heads*2, dim_out, kernel_size=(1,1), bias=bias)
    def forward(self, x, xt):
        '''
        Args:
            x: b,n,di
            x_1:b,n,di
        Returns:
            x: b,n,do
        '''
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x,xt) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(1)
        x = F.elu(self.output(x))
        x = x.squeeze(1)
        return x

class EAGNN(nn.Module):
    def __init__(
            self,
            num_nodes,
            input_dim,
            output_dim,
            in_steps,
            out_steps,
            dim_embed,
            rnn_units,
            rnn_layers,
            dim_embed_feature=120,

    ):
        super(EAGNN, self).__init__()
        # encoder
        self.encoder = Encoder(num_nodes,dim_embed_feature=dim_embed_feature,input_dim=input_dim)
        self.dstgnn = DSTGNN(num_nodes,dim_embed_feature+input_dim,rnn_units,output_dim,in_steps,out_steps,rnn_layers,dim_embed)
    def forward(self,x):
        # 进行encoding
        enc = self.encoder(x)
        dat = torch.cat((x,enc),dim=-1)
        out = self.dstgnn(dat)
        return out

if __name__ == "__main__":
    # 测试GlobalTime
    # encoder = Encoder(307,4)
    dstgnn = DSTGNN(307,4,32,1,12,12,2,6)
    eagnn = EAGNN(307,4,1,12,12,8,32,2)

    summary(eagnn, [64, 12, 307, 4])
