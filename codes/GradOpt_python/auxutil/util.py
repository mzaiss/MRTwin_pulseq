
import numpy
import torch

def prod(shape):
    """Computes product of shape.
    Args:
        shape (tuple or list): shape.
    Returns:
        Product.
    """
    return numpy.prod(shape)

def _expand_shapes(*shapes):

    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)

def resize(input, oshape, ishift=None, oshift=None,device='cuda'):
    ishape_exp, oshape_exp = _expand_shapes(input.shape, oshape)

    if ishape_exp == oshape_exp:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    copy_shape = [min(i - si, o - so) for i, si, o,
                  so in zip(ishape_exp, ishift, oshape_exp, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape_exp, dtype=input.dtype, device=device)
    input = input.reshape(ishape_exp)
    output[oslice] = input[islice]

    return output.reshape(oshape)