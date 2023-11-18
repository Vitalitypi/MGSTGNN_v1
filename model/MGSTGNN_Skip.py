import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F

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

class MGSTGNN(nn.Module):
    def __init__(
            self,
            st_adj,                    # 邻接矩阵
            st_dis,                    # 距离矩阵
            num_nodes,              #节点数
            input_dim,              #输入维度
            rnn_units,              #GRU循环单元数
            output_dim,             #输出维度
            num_layers,             #GRU的层数
            embed_dim,              #GNN嵌入维度
            in_steps=12,               #输入的时间长度
            out_steps=12,              #预测的时间长度
            predict_time=2,
            gat_hidden=256,
            mlp_hidden=256,
            gat_drop=0.6,
            gat_heads=1,
            gat_alpha=0.2,
            gat_concat=True,
            mlp_act=nn.GELU,
            mlp_drop=.0,
            num_gat=0,
            num_back=0
    ):
        super(MGSTGNN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_grus=[1,2]
        self.encoder = DSTRNN(num_nodes, input_dim, rnn_units, embed_dim, num_layers, in_steps, num_back=num_back,conv_steps=predict_time,
                              num_grus=self.num_grus)

        self.norm = nn.LayerNorm(rnn_units*len(self.num_grus), eps=1e-12)
        self.out_dropout = nn.Dropout(0.1)

        self.end_conv = nn.Conv2d(predict_time, out_steps * self.output_dim, kernel_size=(1, rnn_units*len(self.num_grus)), bias=True)

        self.predict_time = predict_time
    def forward(self, source):
        b,t,n,d = source.size()
        # source: B, T, N, D
        init_state = self.encoder.init_hidden(source.shape[0])#,self.num_node,self.hidden_dim
        _, output = self.encoder(source, init_state) # B, T, N, hidden

        output = self.out_dropout(self.norm(output[:, -self.predict_time:, :, :])) # B, r, N, hidden

        # CNN based predictor
        output = self.end_conv((output)) # B, T*C, N, d'
        output = output.squeeze(-1).reshape(-1, self.out_steps, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2) # B, T, N, C


        return output

class DSTRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, embed_dim, num_layers=1, in_steps=12,
                 num_back=0, conv_steps=2, num_grus=None, conv_bias=True):
        super(DSTRNN, self).__init__()
        assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.num_gru = num_layers - 1
        self.dim_out = dim_out
        self.num_back = num_back
        self.num_grus = num_grus
        self.node_embeddings = nn.Parameter(torch.randn(node_num, embed_dim), requires_grad=True)
        self.time_embeddings = nn.Parameter(torch.randn(in_steps, embed_dim), requires_grad=True)
        self.grus = nn.ModuleList([
            GRUCell(node_num, dim_in, dim_out, embed_dim)
            for _ in range(sum(num_grus))
        ])
        if num_back>0:
            self.backs = nn.ModuleList([
                nn.Linear(dim_out,dim_out)
                for _ in range(sum(num_grus))
            ])
        # predict output
        # self.predictors = nn.ModuleList([
        #     nn.Linear(dim_out,dim_cat)
        #     # nn.Conv2d(conv_steps, 1 * in_steps, kernel_size=(1,dim_out), bias=conv_bias)
        #     for _ in num_grus
        # ])
        # skip
        self.skips = nn.ModuleList([
            # nn.Linear(dim_out,dim_in)
            nn.Conv2d(conv_steps, dim_in * in_steps, kernel_size=(1,dim_out), bias=conv_bias)
            for _ in range(len(num_grus)-1)
        ])
        # norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim_in)
            for _ in range(len(num_grus)-1)
        ])
        # dropout
        self.dropouts = nn.Dropout(p=0.1)
        self.conv_steps = conv_steps
    def forward(self, x, init_state):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
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
                if self.num_back > 0:
                    inp = inp - self.backs[index1+index2](prev_state)
                init_hidden_states[index1+index2] = self.grus[index1+index2](inp, prev_state, self.node_embeddings, self.time_embeddings[t]) # [B, N, hidden_dim]
                inner_states.append(init_hidden_states[index1+index2])
            index1 += self.num_grus[i]
            current_inputs = torch.stack(inner_states, dim=1) # [B, T, N, D]
            outputs.append(current_inputs)
            current_inputs = self.dropouts(current_inputs[:, -self.conv_steps:, :, :])
            if i < len(self.num_grus)-1:
                current_inputs = self.skips[i](current_inputs).reshape(batch_size,time_stamps,node_num,-1)+skip

        predict = torch.cat(outputs,dim=-1)
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
    def forward(self, x, state, node_embeddings, time_embeddings):

        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        input_and_state = torch.cat((x, state), dim=-1) # [B, N, 1+D]
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings,time_embeddings))
        z,r = torch.split(z_r,self.hidden_dim,dim=-1)
        # r = torch.sigmoid(self.gate_r(input_and_state, node_embeddings,time_embeddings))
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
        self.norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.drop = nn.Dropout(0.1)
    def forward(self, x, node_embeddings, time_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D], embedding shaped [N, N]
        # output shape [B, N, C]
        I = torch.eye(self.node_num).to(x.device)
        node_embeddings = self.drop(
            self.norm(node_embeddings + time_embeddings.unsqueeze(0)))  # torch.mul(node_embeddings, node_time)
        embedding = F.softmax(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)), dim=1)
        support_set = [I, embedding]
        supports = torch.stack(support_set, dim=0)  # [3, N, N]
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) # N, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool) # N, dim_out

        x_g = torch.einsum("knm,bmc->bknc", supports, x) # B, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias # B, N, dim_out
        return x_gconv


class Network(nn.Module):
    def __init__(
            self,
            st_adj,
            st_dis,
            num_nodes,
            input_dim,
            output_dim,
            in_steps,
            out_steps,
            dim_embed,
            rnn_units,
            rnn_layers,

            # Encoder configuration
            input_embedding_dim,
            periods_embedding_dim,
            weekend_embedding_dim,
            holiday_embedding_dim,
            spatial_embedding_dim,
            adaptive_embedding_dim,
            dim_embed_feature,
            periods,

            predict_time,
            gat_hidden,
            mlp_hidden,
            gat_drop=0.6,
            gat_heads=1,
            gat_alpha=0.2,
            gat_concat=True,
            mlp_act='gelu',
            mlp_drop=.0,
            num_gat=0,
            num_back=0
    ):
        super(Network, self).__init__()
        if mlp_act=='gelu':
            mlp_act = nn.GELU
        # encoder
        # self.encoder = Encoder(num_nodes,input_embedding_dim,periods_embedding_dim,weekend_embedding_dim,
        #                        holiday_embedding_dim,spatial_embedding_dim,adaptive_embedding_dim,
        #                        dim_embed_feature,input_dim,periods)
        self.mgstgnn = MGSTGNN(st_adj,st_dis,num_nodes,input_dim,rnn_units,output_dim,
                               rnn_layers,dim_embed,in_steps=in_steps,out_steps=out_steps,predict_time=predict_time,
                               gat_hidden=gat_hidden, mlp_hidden=mlp_hidden,gat_drop=gat_drop,gat_heads=gat_heads,
                               gat_alpha=gat_alpha,gat_concat=gat_concat, mlp_act=mlp_act, mlp_drop=mlp_drop,
                               num_gat=num_gat,num_back=num_back)
    def forward(self,x):
        # 进行encoding

        # enc = self.encoder(x)

        # dat = torch.cat((x,enc),dim=-1)
        out = self.mgstgnn(x)
        return out

if __name__ == "__main__":
    periods_dim = [24]
    periods_arr = [288]
    num_nodes = 307
    device = torch.device("cuda", 0)
    st_adj = torch.randn(num_nodes*2,num_nodes*2).to(device)
    st_dis = torch.randn(num_nodes*2,num_nodes*2).to(device)
    network = Network(st_adj,st_dis,307,4,1,12,12,8,32,3,80,periods_dim,6,2,0,8,120,periods_arr,2, 256, 256,
                      0.6, 1, 0.2, True, 'gelu', .0, 0)

    summary(network, [64, 12, 307, 4])
