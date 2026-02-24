import argparse
import torch
import torch.nn.functional as F
import yaml
import torch.nn as nn

parser = argparse.ArgumentParser("HSI")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--path-config', type=str, default=r'./config/config.yaml')
args = parser.parse_args()

config = '/home/JiLiang/GCGV/config/config.yaml'
with open(config, 'r', encoding='utf-8') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
alfa = configs["weight parameter"]["alfa"]
beta = configs["weight parameter"]["beta"]


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Feature_Extract(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(Feature_Extract, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class CAM(nn.Module):
    """ Channel Attention Module"""

    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class SAM(nn.Module):
    """ Spatial Attention Module"""
    
    def __init__(self, in_dim):
        super(SAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class GAT_Layer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GAT_Layer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha

        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._Encoder(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _Encoder(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, adj, nout, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj = adj
        self.attentions = [GAT_Layer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GAT_Layer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return x

class SSConv(nn.Module):
    """Spectral-Spatial Convolution"""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class Branch1(nn.Module):

    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,
                 model='normal'):
        super(Branch1, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        layers_count = 2
        self.channel_attention_1 = ChannelAttention(128)
        self.spatial_attention_1 = SpatialAttention(kernel_size=7)
        self.channel_attention_2 = ChannelAttention(64)
        self.spatial_attention_2 = SpatialAttention(kernel_size=7)
        self.WH = 0
        self.M = 2
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_Network = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.CNN_Network.add_module('Attention' + str(i), SAM(128))
                self.CNN_Network.add_module('Attention' + str(i), CAM(128))
                self.CNN_Network.add_module('CNN_Network' + str(i), SSConv(128, 128, kernel_size=3))

            else:
                self.CNN_Network.add_module('Attention' + str(i), SAM(128))
                self.CNN_Network.add_module('Attention' + str(i), CAM(128))
                self.CNN_Network.add_module('CNN_Network' + str(i), SSConv(128, 64, kernel_size=5))

        self.GAT_Network = nn.Sequential()
        self.GAT_Network.add_module('GAT_Network' + str(i),
                                   GAT(nfeat=128, nhid=30, adj=A, nout=64, dropout=0.4, nheads=4, alpha=0.2))
        self.linear1 = nn.Linear(64, 64)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise
        clean_x_flatten = clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)
        hx = clean_x
        CNN_Out = self.CNN_Network(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # [21025,64]
        CNN_Out = torch.squeeze(CNN_Out, 0).permute([1, 2, 0]).reshape([h * w, -1])
        H = superpixels_flatten
        H = self.GAT_Network(H)
        GAT_Out = torch.matmul(self.Q, H) # [21025,64]
        GAT_Out = self.linear1(GAT_Out)
        GAT_Out = self.act1(self.bn1(GAT_Out))

        # Y  = 0.06 * CNN_Out + 0.94 * GAT_Out
        Y = alfa * CNN_Out + beta * GAT_Out  # [21025,64]

        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)  # [21025,16]

        # print(Y.shape)
        return Y




























# class ViT(nn.Module):
#     def __init__(self, n_gcn, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=16,
#                  dropout=0., emb_dropout=0., mode='CAF'):
#         super().__init__()
#
#         patch_dim = n_gcn
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.dropout = nn.Dropout(emb_dropout)
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#
#     def forward(self, x, mask=None):
#         x = x.to(torch.float32)
#
#         x = self.patch_to_embedding(x)
#         b, n, _ = x.shape
#
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         pos = self.pos_embedding[:, :(n + 1)]
#         x += pos
#         x = self.dropout(x)
#
#         x = self.transformer(x, mask)
#
#         x = self.to_latent(x[:, 0])
#         x = self.mlp_head(x)
#
#         return x
#