import torch
import numpy as np


class Splitter(torch.nn.Module):
    def __init__(self, aug, p):
        super().__init__()
        self.aug = aug
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            n = x.shape[0]
            if n == 1:
                return self.aug(x)
            tt = torch.chunk(x, n//2)
            tas = [self.aug(t) for t in tt]
            x = torch.vstack(tas)
        return x


class NoiseInjection(torch.nn.Module):
    def __init__(self, max_noise_level, p):
        super().__init__()
        self.noise_level = (0.0, max_noise_level)
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            noise_level = np.random.uniform(*self.noise_level)
            noise = torch.randn_like(x)
            x = (x + noise * noise_level)
        return x


class HDRAug(torch.nn.Module):
    def __init__(self, max_shift, p):
        super().__init__()
        self.MAXH = self.MAXW = max_shift # pixshift
        self.p = p

    @torch.no_grad()
    def forward(self, x, y):
        if torch.rand(1) < self.p:
            x, y = hdr(x,y, self.MAXW, self.MAXH, 3) # 3y per 1x
        return x, y


def hdr(xb, yb, max_w, max_h, chan_mult=3):
    # B,3,H,W
    _, C, _, _ = xb.shape
    xchans = torch.split(xb, 1, dim=1)
    ychans = torch.split(yb, chan_mult, dim=1)

    xshifted = []
    yshifted = []
    to_shift = torch.randint(0,C,(1,))[0]
    for i in range(len(xchans)):
        xch = xchans[i]
        ych = ychans[i]
        if i == to_shift:
            shifts = (torch.randint(1, max_h, (1,)),torch.randint(1, max_w, (1,)))
            xch = torch.roll(xch, shifts=shifts, dims=(2,3))
            ych = torch.roll(ych, shifts=shifts, dims=(2,3))
        xshifted.append(xch)
        yshifted.append(ych)

    xb = torch.hstack(xshifted)
    yb = torch.hstack(yshifted)
    return xb, yb
