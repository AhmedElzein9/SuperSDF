import torch
from torch import nn
import math
from einops import rearrange
from mamba_simple import Mamba


def split_band(x, move_num, spec_num):
    """
    输入: x (Tensor) - 形状为 [b, c] 的张量
    move_num: 每次移动的步长
    spec_num: 每次切片的大小
    返回: 形状为 [b, n, c] 的张量，其中 n 是根据 move_num 和 spec_num 计算得到的切片数量
    """
    b, c, h, w = x.shape
    slices = []
    for i in range(0, c, move_num):
        if i + spec_num > c:
            slice = x[:, c - spec_num:c, :, :]  # 取最后 spec_num 列
        else:
            slice = x[:, i:i + spec_num, :, :]
        slices.append(slice)
    # 使用 torch.stack 而不是逐个转换到 CPU 再转回 CUDA，提高效率
    slices = torch.stack(slices, dim=1)  # 将切片堆叠成新的维度
    return slices


class Residual_SSMN(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# 等于 PreNorm
class LayerNormalize_SSMN(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class mamba_block1(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Residual_SSMN(LayerNormalize_SSMN(dim, Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=dim,  # Model dimension d_model
                    d_state=64,  # SSM state expansion factor # 64
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    use_fast_path=False,
                )))
            )

    def forward(self, x):
        for attention in self.layers:
            x = attention(x)
        return x


class mamba_block2(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Residual_SSMN(LayerNormalize_SSMN(dim, Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=dim,  # Model dimension d_model
                    d_state=64,  # SSM state expansion factor # 64
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    use_fast_path=False,
                )))
            )

    def forward(self, x):
        for attention in self.layers:
            x = attention(x)
        return x


class mambaLG(nn.Module):
    def __init__(self, num_classes=16, dim=64, depth=1, dropout=0.1, band=30, spec_num=12, spec_rate=0.5, device='0',
                 spa_token=16):
        super(mambaLG, self).__init__()
        self.name = 'mambaLG'
        self.spec_num = spec_num
        self.spec_rate = spec_rate
        self.move_num = int(math.ceil(self.spec_num * self.spec_rate))
        self.device = device
        # dim = band

        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # # 空间分支
        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
            # nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_features2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.conv2d_channel = nn.Sequential(
            nn.Conv2d(dim, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_fusion = nn.Sequential(
            nn.Conv2d(4 * dim, out_channels=dim, kernel_size=(1, 1)),
            # nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

        self.SPAM = mamba_block1(dim, depth)  # 原版

        self.spa_token = spa_token
        self.localSPAM = mamba_block1(dim, depth)

        spe_dim = dim
        self.nn1 = nn.Sequential(
            nn.Linear(dim, spe_dim),
            nn.LayerNorm(spe_dim),
            nn.GELU(),
        )  #
        dim = spe_dim

        # 光谱分支
        num_patch = math.floor((dim - (self.spec_num - self.move_num)) / self.move_num) + \
                    math.ceil(
                        (((dim - (self.spec_num - self.move_num)) % self.move_num) + (self.spec_num - self.move_num)) / \
                        self.move_num)

        self.spe_token1 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.spe_token2 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.SPEM = mamba_block2(self.spec_num, depth)

        self.nn2 = nn.Sequential(
            nn.Linear(self.spec_num * num_patch, dim),  # 原始光谱
            nn.LayerNorm(dim),
            nn.GELU(),
        )  #
        self.outhead = nn.Sequential(
            # nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )

    def forward(self, x, test=False):
        B, H, W, C = x.shape
        x = self.preprocess(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # # 先空间后光谱
        x_spe = x
        # spatial
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv2d_channel(x)
        x2 = self.conv2d_features1(x)
        x3 = self.conv2d_features2(x)
        # x = x1 + x2 + x3 + x
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv2d_fusion(x)
        x = self.dropout(x)  # .permute(0, 2, 3, 1)
        x_s1 = x
        # expand H and W
        eH = self.spa_token - H % self.spa_token
        eW = self.spa_token - W % self.spa_token
        pad = torch.nn.ReflectionPad2d((0, eW, 0, eH)).to(self.device)
        x = pad(x)
        bb, cc, hh, ww = x.shape
        x = rearrange(x, 'b c (nh htoken) (nw wtoken)-> (b nh nw) (wtoken htoken) c', htoken=self.spa_token,
                      wtoken=self.spa_token)
        x = self.localSPAM(x)
        x = rearrange(x, '(b nh nw) (wtoken htoken) c-> b (nh htoken) (nw wtoken) c', htoken=self.spa_token,
                      wtoken=self.spa_token, nh=hh // self.spa_token, nw=ww // self.spa_token)
        x = x[:, 0:H, 0:W, :] + x_s1.permute(0, 2, 3, 1)

        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.SPAM(x)
        x = rearrange(x, 'b (h w) c-> b h w c', h=H, w=W)

        x_spa = self.nn1(x)

        # # # 光谱
        x_s = x_spe.unsqueeze(1)
        x_s = self.spe_token1(x_s) + self.spe_token2(x_s) + x_s
        x_s = self.dropout(x_s).squeeze(1).permute(0, 3, 1, 2)
        Patch_pool = torch.nn.AvgPool2d((H, W)).cuda()
        x_s = Patch_pool(x_s)
        x_s = split_band(x_s, self.move_num, self.spec_num)
        bb, nn, cc, hh, ww = x_s.shape
        x_s = rearrange(x_s, 'b n c h w-> (b h w) n c')
        x_s = self.SPEM(x_s)
        x_s = rearrange(x_s, '(b h w) n c-> b (n c) (h w)', h=hh, w=ww).mean(-1)
        x_s = self.nn2(x_s).unsqueeze(1).unsqueeze(1)
        x = x_spa * x_s  # +x_s2  # + x_s2
        xout = x
        x = rearrange(x, 'b h w c-> b c h w')
        x = self.outhead(x)

        if test is True:
            return x, xout, x_s
        else:
            return x, x_s
