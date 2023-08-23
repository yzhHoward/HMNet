import math
import torch
from torch import nn
import torch.nn.functional as F
from faiss import StandardGpuResources
from faiss import METRIC_L2
from faiss.contrib.torch_utils import torch_replacement_knn_gpu

from layers.Embed import TemporalEmbedding


class CustomEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(CustomEmbedding, self).__init__()
        self.temporal_embedding = TemporalEmbedding(d_model=d_model)
        self.start_fc = nn.Linear(c_in, d_model)

    def forward(self, x, x_mark):
        x = self.start_fc(x.unsqueeze(-1)) + self.temporal_embedding(x_mark).unsqueeze(-2)# + self.position_embedding(x).unsqueeze(-2)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        input_dim = configs.enc_in
        output_dim = configs.c_out
        lag = configs.seq_len
        horizon = configs.pred_len
        hidden_dim = 32
        device = torch.device('cuda:0')
        print('Predicting {} steps ahead'.format(horizon))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding = CustomEmbedding(1, hidden_dim)
        self.layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.horizon = horizon
        self.lag = lag
        self.res = StandardGpuResources()

        patch_sizes = (6, 4, 4)
        
        if configs.task_id == 'ETTh1':
            proto = (False, True, True)
            # feature_interaction = (False, False, False)
            feature_interaction = (False, True, True)
        elif configs.task_id == 'ETTh2':
            proto = (False, True, True)
            feature_interaction = (False, True, True)
        elif configs.task_id == 'ETTm1':
            proto = (False, False, False)
            feature_interaction = (False, True, True)
        elif configs.task_id == 'ETTm2':
            proto = (False, True, True)
            feature_interaction = (False, True, True)
        elif configs.task_id == 'ECL':
            proto = (False, True, True)
            feature_interaction = (False, True, True)
        elif configs.task_id == 'Exchange':
            proto = (True, False, False)
            feature_interaction = (False, False, False)
        elif configs.task_id == 'traffic':
            proto = (False, True, True)
            feature_interaction = (True, True, True)
        elif configs.task_id == 'weather':
            proto = (True, True, True)
            feature_interaction = (True, True, True)
        elif configs.task_id == 'ili':
            proto = (True, True, False)
            feature_interaction = (False, False, False)
        else:
            proto = (False, True, True)
            feature_interaction = (False, True, True)

        cuts = lag

        for patch_size, p, f in zip(patch_sizes, proto, feature_interaction):
            if cuts % patch_size != 0:
                raise Exception('Lag not divisible by patch size')

            cuts = int(cuts / patch_size)
            self.layers.append(Layer(device=device, input_dim=input_dim, hidden_dim=hidden_dim, cuts=cuts,
                                     cut_size=patch_size, proto=p, res=self.res, interaction=f, memsize=configs.mem_size, k=configs.k))
            self.residual_layers.append(nn.Linear(cuts * hidden_dim, 256))

        self.projections = nn.Sequential(*[
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.horizon)])

        self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))

    def forward(self, x, x_mark, x_dec, x_mark_dec, return_h = False):
        # b, t, d; b, t, d1
        batch_size = x.size(0)

        mean = torch.mean(x, dim=1, keepdim=True).detach() #.repeat(1, self.horizon, 1)]
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - mean) / std
        x = x * self.affine_weight + self.affine_bias

        x = self.embedding(x, x_mark)
        skip = 0
        hs = []

        x = x.permute(0, 2, 3, 1)  # b, d, h, t
        for layer, residual_layer in zip(self.layers, self.residual_layers):
            x, h = layer(x)  # b, c, d, h
            hs.append(h)
            skip_inp = x.reshape(batch_size, self.input_dim, -1)  # b, d, h
            skip = skip + residual_layer(skip_inp)

        x = self.projections(skip).transpose(2, 1)

        x = (x - self.affine_bias) / (self.affine_weight + 1e-10) * std + mean
        if return_h:
            return x, hs
        else:
            return x


class FeatureRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureRegression, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.Tensor(hidden_dim))
        m = (torch.ones(input_dim, input_dim) - torch.eye(input_dim, input_dim)).repeat_interleave(hidden_dim // input_dim, dim=0).repeat_interleave(hidden_dim // input_dim, dim=1)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * self.m, self.b)
        return z_h


class Layer(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, cuts, cut_size, proto=False, res=None, interaction=False, memsize=4096, k=16):
        super(Layer, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.cuts = cuts
        self.cut_size = cut_size

        self.conv = nn.Conv1d(hidden_dim, hidden_dim, cut_size, cut_size)
        self.interaction = interaction
        if interaction:
            self.feature_interaction = FeatureRegression(input_dim, input_dim)
            self.feature_W = nn.Linear(input_dim * 2, input_dim)

        self.dropout = nn.Dropout(0.1)

        self.proto = proto
        if proto is True:
            self.k = k
            self.res = res
            self.mem_size = memsize
            self.mem_cnt = 0
            self.memory = nn.Parameter(torch.rand((self.mem_size, input_dim * hidden_dim), dtype=torch.float32, device='cuda'), requires_grad=False)
            self.similar_att = SimilarityAttention(hidden_dim)
            self.memory_W = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # x shape: B T N C
        batch_size = x.size(0)

        out1 = None

        if self.interaction:
            x = x.permute(0, 2, 3, 1)
            x1 = self.feature_interaction(x)
            beta = torch.sigmoid(self.feature_W(torch.cat((x, x1), dim=-1)))
            x = beta * x + (1 - beta) * x1
            x = x.permute(0, 3, 1, 2)

        x = x.reshape(batch_size * self.input_dim, self.hidden_dim, -1)
        x = self.conv(x)
        x = x.reshape(batch_size, self.input_dim, self.hidden_dim, -1)
        if self.proto:
            x = x.permute(0, 3, 1, 2)
            out1 = F.normalize(x.detach().reshape(-1, self.input_dim * self.hidden_dim), dim=-1)
            similar = self.get_similar(out1)
            similar_h, attn = self.similar_att(x, similar.reshape(batch_size, self.cuts, self.k, self.input_dim, self.hidden_dim).transpose(-3, -2))
            alpha = torch.sigmoid(self.memory_W(torch.cat((x, similar_h), dim=-1)))
            x = x * alpha + similar_h * (1 - alpha)
            x = x.permute(0, 2, 3, 1)
        return self.dropout(x), out1

    def get_similar(self, x):
        with torch.no_grad():
            distance, index = torch_replacement_knn_gpu(
                self.res, x, self.memory, self.k, metric=METRIC_L2)
            return self.memory[index]

    def set_memory(self, value):
        with torch.no_grad():
            index = torch.arange(self.mem_cnt, self.mem_cnt + value.shape[0], device='cuda')
            self.mem_cnt += value.shape[0]
            if self.mem_cnt >= self.mem_size:
                self.mem_cnt -= self.mem_size
                index = torch.where(index < self.mem_size, index, index - self.mem_size)
            self.memory.data[index] = value


class SimilarityAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimilarityAttention, self).__init__()
        self.input_dim = input_dim
        hidden_dim = input_dim
        self.hidden_dim = input_dim
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_t, similar):  # b, h; b, k, h
        input_q = self.W_q(h_t)
        input_k = self.W_k(similar)
        input_v = self.W_v(similar)
        # e = -torch.norm(h_t.unsqueeze(-2) - similar, dim=-1)
        # e = torch.cosine_similarity(h_t.unsqueeze(-2).repeat(1, 1, 1, similar.shape[-2], 1).reshape(-1, h_t.shape[-1]), similar.reshape(-1, h_t.shape[-1])).reshape(similar.shape[:-1])
        e = torch.matmul(input_k, input_q.unsqueeze(-1)).squeeze(-1)
        a = self.softmax(e)  # b, k
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(-2), input_v).squeeze(-2)  # b, h
        return v, a
