
import torch
import numpy
from auxutil import util

def interpolate(input, width, kernel, coord, device):
    ndim = coord.shape[-1]

    batch_shape = input.shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    input = input.reshape([batch_size] + list(input.shape[-ndim:]))
    coord = coord.reshape([npts, ndim])
    output = torch.zeros([batch_size, npts], dtype=input.dtype, device=device)

    output = _interpolate2(output, input, width, kernel, coord)

    return output.reshape(batch_shape + pts_shape)


def bilinear_interpolate_torch_gridsample(input, coord):
    coord=coord.unsqueeze(0).unsqueeze(0)
    tmp=torch.zeros_like(coord)
    tmp[:, :, :, 0] = ((coord[:, :, :, 1]+input.shape[2]/2) / (input.shape[2]-1))  # normalize to between  -1 and 1
    tmp[:, :, :, 1] = ((coord[:, :, :, 0]+input.shape[2]/2) / (input.shape[2]-1)) # normalize to between  -1 and 1
    tmp = tmp * 2 - 1  # normalize to between -1 and 1
    tmp=tmp.expand(input.shape[0],-1,-1,-1)
    return torch.nn.functional.grid_sample(input, tmp).squeeze(2)

def lin_interpolate(kernel, x):
    mask=torch.lt(x,1).float()
    x = x.clone()*mask
    n = len(kernel)
    idx = torch.floor(x * n)
    frac = x * n - idx

    left = kernel[idx.long()]
    mask2=torch.ne(idx,n-1).float()
    idx=idx.clone() * mask2
    right = kernel[idx.long() + 1]
    output=(1.0 - frac) * left + frac * right
    return output*mask*mask2


def _interpolate2(output, input, width, kernel, coord):
    batch_size, ny, nx = input.shape

    kx, ky = coord[:, -1], coord[:, -2]
    x0, y0 = (torch.ceil(kx - width / 2),
              torch.ceil(ky - width / 2))
    input = torch.view_as_real(input)
    for y in range(int(width) + 1):
        wy = lin_interpolate(kernel, torch.abs(y0 + y - ky) / (width / 2))

        for x in range(int(width) + 1):
            w = wy * lin_interpolate(kernel, torch.abs(x0 + x - kx) / (width / 2))

            yy = torch.fmod(y0+y, ny).long()
            xx = torch.fmod(x0+x, nx).long()
            output[:, :] = output[:, :].clone() + w * torch.view_as_complex(input[:, yy, xx])

    return output

def gridding(input, shape, width, kernel, coord,device):
    ndim = coord.shape[-1]

    batch_shape = shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    input = input.reshape([batch_size, npts])
    coord = coord.reshape([npts, ndim])
    output = torch.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype, device=device)

    output=_gridding2(output, input, width, kernel, coord)

    return output.reshape(shape)

def _gridding2(output, input, width, kernel, coord):
    batch_size, ny, nx = output.shape

    kx, ky = coord[:, -1], coord[:, -2]

    x0, y0 = (torch.ceil(kx - width / 2),
              torch.ceil(ky - width / 2))

    for y in range(int(width) + 1):
        wy = lin_interpolate(kernel, torch.abs(y0 + y - ky) / (width / 2))

        for x in range(int(width) + 1):
            w = wy * lin_interpolate(kernel, torch.abs(x0 + x - kx) / (width / 2))

            yy = torch.fmod(y0+y,ny).long()
            xx = torch.fmod(x0+x,nx).long()
            # output[:, yy, xx] = output[:, yy, xx] + w * input[:, :]
            update = torch.zeros_like(output)
            update[:, yy, xx] = w * input[:, :]
            output = output + update

    return output
