import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
import scipy.io as sio

class SineLayer(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True, is_first=False, omega_0=30):
        super(SineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_feature = in_feature
        self.linear = nn.Linear(in_feature, out_feature, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_feature, 1 / self.in_feature)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_feature) / self.omega_0,
                                             np.sqrt(6 / self.in_feature) / self.omega_0)

    def forward(self, x):
        tmp = self.linear(x)*self.omega_0
        res = torch.sin(tmp)

        return res


class Sine(nn.Module):
    def __init__(self, log=False):
        super(Sine, self).__init__()
        self.log = log

    def forward(self, x):
        if self.log:
            print(torch.sin(x))
        return torch.sin(x)*x


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
            m.weight *= 30
def save_mat(net, name):
    mat = {}
    for i, m in enumerate(net):
        if isinstance(m, SineLayer):
            mat[f'layer_{i}'] = m.linear.weight.cpu().detach().numpy()
        if isinstance(m, nn.Linear):
            mat[f'layer_{i}'] = m.weight.cpu().detach().numpy()
    sio.savemat(name, mat)

gt = ToTensor()(Image.open('butterfly.png')).cuda()

writer = SummaryWriter()
psnr = PeakSignalNoiseRatio(data_range=1)
network = nn.Sequential(
    SineLayer(2, 256, is_first=True),
    SineLayer(256, 256),
    SineLayer(256, 256),
    SineLayer(256, 256),
    nn.Linear(256, 3),
).cuda()

optimizer = Adam(network.parameters(), lr=1e-4, weight_decay=1e-5)

grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, gt.shape[1]), torch.linspace(0, 1, gt.shape[2]), indexing='xy'),
                   dim=-1) \
    .reshape(-1, 2).cuda()

for epoch in range(5000):
    optimizer.zero_grad()
    pred = network(grid)
    loss1 = F.mse_loss(pred, einops.rearrange(gt, 'c h w -> (h w) c'))
    loss = loss1
    loss.backward()
    optimizer.step()
    out = einops.rearrange(pred, "(h w) c -> c h w", h=gt.shape[1])
    print(f"epoch:{epoch}:{loss.item()} PSNR:{psnr(out.cpu(), gt.cpu())}")

    if epoch % 500 == 0:
        ToPILImage()(einops.rearrange(pred, "(h w) c -> c h w", h=gt.shape[1])).save(
            f'./results/image_representation/{epoch}.jpg')
    if (epoch + 1) % 1000 == 0:
        torch.save(network.state_dict(), f'./results/{epoch + 1}.pth')
