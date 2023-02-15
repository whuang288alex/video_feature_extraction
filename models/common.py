import torch
from torch.nn import Module


class FeedVideoInput(Module):
    
    def __init__(self, model: Module):
        super().__init__()
        
        self.model = model

    def forward(self, x) -> torch.Tensor:
        input = x['video']
        if input.ndim == 5:     # (bs, 3, n, h, w)
            out = self.model(input)
        elif input.ndim == 6:   # (bs, c, 3, n, h, w)
            c = input.size(1)
            # evaluate one crop at a time
            out = [self.model(input[:, i]) for i in range(c)]
            out = torch.stack(out, dim=-1)
        else:
            raise ValueError('invalid input size')
        return out


class FeedVideoInputList(Module):

    def __init__(self, model: Module):
        super().__init__()

        self.model = model

    def forward(self, x) -> torch.Tensor:
        input = x['video']
        assert isinstance(input, (list, tuple))
        
        if input[0].ndim == 5:
            out = self.model(input)
        elif input[0].ndim == 6:
            # bs, crop, channel, T, w, h
            c = input[0].size(1)
            out = [self.model([x[:, i] for x in input]) for i in range(c)]
            out = torch.stack(out, dim=-1)
        else:
            raise ValueError('invalid input size')
        return out


class Mirror:
    def __call__(self, im):
        if im.ndim == 4:
            im = torch.stack([im, torch.flip(im, dims=(-1,))])
        elif im.ndim == 5:
            im = torch.cat([im, torch.flip(im, dims=(-1,))])
        else:
            raise ValueError()
        return im


class ThreeCrop:

    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        h, w = im.shape[-2:]
        assert h >= self.size and w >= self.size

        y0 = h // 2 - self.size // 2
        y1 = h // 2 + self.size // 2

        # middle crop
        x0_m = w // 2 - self.size // 2
        x1_m = w // 2 + self.size // 2
        im_m = im[..., x0_m:x1_m]
        # left crop
        im_l = im[..., :self.size]
        # right crop
        im_r = im[..., -self.size:]

        im = torch.stack([im_l, im_m, im_r])

        return im