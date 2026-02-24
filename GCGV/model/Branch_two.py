import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import timm
import torchvision.models as models
from einops import rearrange, repeat


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='HSI')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--path-config', type=str, default=r'./config/config.yaml')
args = parser.parse_args()

config = '/home/JiLiang/GCGV/config/config.yaml'
with open(config, 'r', encoding='utf-8') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
alfa = configs["weight parameter"]["alfa"]
beta = configs["weight parameter"]["beta"]

# models = timm.list_models()
# print(models)


class attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask error'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij, bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


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
    def __init__(self, kernel_size=3):
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


class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Res_Cnt(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class GCN_Layer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCN_Layer, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))

    def A2D(self, A: torch.Tensor):
        D = A.sum(2)
        batch, l = D.shape
        D1 = torch.reshape(D, (batch * l, 1))
        D1 = D1.squeeze(1)
        D2 = torch.pow(D1, -0.5)
        D2 = torch.reshape(D2, (batch, l))
        D_hat = torch.zeros([batch, l, l], dtype=torch.float)
        for i in range(batch):
            D_hat[i] = torch.diag(D2[i])
        return D_hat.cuda()

    def A2D_2(self, A: torch.Tensor):
        D = A.sum(2)
        batch, l = D.shape
        D1 = torch.reshape(D, (batch * l, 1))
        D1 = D1.squeeze(1)
        D2 = torch.pow(D1, 0.5)
        D2 = torch.reshape(D2, (batch, l))
        D_hat = torch.zeros([batch, l, l], dtype=torch.float)
        for i in range(batch):
            D_hat[i] = torch.diag(D2[i])
        return D_hat.cuda()

    def forward(self, H, A):
        nodes_count = A.shape[1]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        (batch, l, c) = H.shape
        H1 = torch.reshape(H, (batch * l, c))
        H2 = self.BN(H1)
        H = torch.reshape(H2, (batch, l, c))
        D_hat = self.A2D(A)
        D_hat2 = self.A2D_2(A)

        # Ln = I - D^ A D^
        A_hat = torch.matmul(D_hat2, torch.matmul(A, D_hat))
        A_hat = I - A_hat
        output = torch.matmul(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)
        return output


class GCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int):
        super(GCN, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        layers_count = 4
        self.GCN_Network = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                if i == 0:
                    self.GCN_Network.add_module('GCN_Network' + str(i), GCN_Layer(self.channel, 128))
                else:
                    self.GCN_Network.add_module('GCN_Network' + str(i), GCN_Layer(128, 128))
            else:
                self.GCN_Network.add_module('GCN_Network' + str(i), GCN_Layer(128, 64))
        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention(kernel_size=3)
        self.BN = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor, A: torch.Tensor, indexs_train):
        (batch, h, w, c) = x.shape
        _, in_num = indexs_train.shape
        H = torch.reshape(x, (batch, h * w, c))
        for i in range(len(self.GCN_Network)):
            H = self.GCN_Network[i](H, A)

        _, _, c_gcn = H.shape
        gcn_out = torch.zeros((batch, in_num, c_gcn), dtype=float)
        gcn_out = gcn_out.type(torch.cuda.FloatTensor)
        for i in range(batch):
            gcn_out[i] = H[i][indexs_train[i]]

        gcn_out = gcn_out.transpose(1, 2)
        gcn_out = gcn_out.unsqueeze(3)
        gcn_out = self.ca(gcn_out) * gcn_out
        gcn_out = self.sa(gcn_out) * gcn_out

        gcn_out = gcn_out.squeeze(3)
        gcn_out = self.BN(gcn_out)
        gcn_out = gcn_out.transpose(1, 2)
        tr_in = gcn_out.transpose(1, 2)
        return tr_in.cuda()


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Res_Cnt(Norm(dim, attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Res_Cnt(Norm(dim, MLP_Block(dim, mlp_head, dropout=dropout)))
            ]))
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        last_output = []
        nl = 0
        for attn, ff in self.layers:
            last_output.append(x)
            if nl > 1:
                x = self.skipcat[nl - 2](
                    torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
            x = attn(x, mask=mask)
            x = ff(x)
            nl += 1
        return x


class ViT(nn.Module):
    def __init__(self, n_gcn, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=16,
                 dropout=0., emb_dropout=0. ):
        super().__init__()
        patch_dim = n_gcn
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        x = x.to(torch.float32)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        pos = self.pos_embedding[:, :(n + 1)]
        x += pos
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_latent(x[:, 0])
        x = self.mlp_head(x)
        #print(x.shape)
        return x

