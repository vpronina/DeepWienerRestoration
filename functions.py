import numpy as np
import math
import torch
import torch.nn as nn
import utils
import random
import torch.nn.functional as F
from networks import UNet
from utils import cmul, cabs, conj


class Wiener(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, blurKernel, weights, alpha):
        """
        Wiener Filter for a batch of input images. (Filtering is taking place in the Frequency domain under the
                 assumption of periodic boundary conditions for the input image.)

        input: (torch.(cuda.)Tensor) Input image tensor of size B x C x H x W
        blurKernel: (torch.(cuda.)Tensor) PSFs tensor of size B x C x Hk x Wk
        weights: (torch.(cuda.)Tensor) Regularization kernels of size D x C x Hw x Ww
        alpha: (float) Regularization parameter of shape 1 x 1
        returns: (torch.(cuda.)Tensor) Wiener filter output tensor B x 1 x C x H x H

        output = F^H (B^H*F(input)/(|B|^2+exp(alpha)*|W|^2))
        """

        assert (input.dim() < 5), "The input must be at most a 4D tensor."
        while input.dim() < 4:
            input = input.unsqueeze(0)

        batch = input.size(0)
        channels = input.size(1)

        assert (blurKernel.dim() < 5), "The blurring kernel must be at most a 4D tensor."
        while blurKernel.dim() < 4:
            blurKernel = blurKernel.unsqueeze(0)

        bshape = tuple(blurKernel.shape)
        assert (bshape[0] in (1, batch) and bshape[1] in (1, channels)), "Invalid blurring kernel dimensions."

        N = alpha.size(0)
        assert (alpha.dim() == 2 and alpha.size(-1) in (1, channels)), \
            "Invalid dimensions for the alpha parameter. The expected shape of the " \
            + "tensor is {} x [{}|{}]".format(N, 1, channels)
        alpha = alpha.exp()

        assert (weights.dim() > 3 and weights.dim() < 6), "The regularization " \
                                                          + "kernel must be a 4D or 5D tensor."

        if weights.dim() < 5:
            weights = weights.unsqueeze(0)

        wshape = tuple(weights.shape)
        assert (wshape[0] in (1, N) and wshape[2] in (1, channels)), \
            "Invalid regularization kernel dimensions."

        # Zero-padding of the blur kernel to match the input size
        B = torch.zeros(bshape[0], bshape[1], input.size(2), input.size(3)).type_as(blurKernel)
        B[..., 0:bshape[2], 0:bshape[3]] = blurKernel
        del blurKernel
        # Circular shift of the zero-padded blur kernel
        bs = tuple(int(i) for i in -(np.asarray(bshape[-2:]) // 2))
        bs = (0, 0) + bs
        B = utils.shift(B, bs, bc='circular')
        # FFT of B
        B = torch.rfft(B, 2) 

        # Zero-padding of the spatial dimensions of the weights to match the input size
        G = torch.zeros(wshape[0], wshape[1], wshape[2], input.size(2), input.size(3)).type_as(weights)
        G[..., 0:wshape[3], 0:wshape[4]] = weights
        del weights
        # circular shift of the zero-padded weights
        ws = tuple(int(i) for i in -(np.asarray(wshape[-2:]) // 2))
        ws = (0, 0, 0) + ws
        G = utils.shift(G, ws, bc='circular')
        # FFT of G
        G = torch.rfft(G, 2) 

        Y = cmul(conj(B), torch.rfft(input, 2)).unsqueeze(1) 

        ctx.intermediate_results = tuple()
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            ctx.intermediate_results += (alpha, B, G, Y, wshape)
        elif ctx.needs_input_grad[0]:
            ctx.intermediate_results += (alpha, B, G)

        B = cabs(B).unsqueeze(-1) 
        G = cabs(G).pow(2).sum(dim=1)  

        G = G.mul(alpha.unsqueeze(-1).unsqueeze(-1)).unsqueeze(0).unsqueeze(-1) 

        G = G + B.pow(2).unsqueeze(1) 
        return torch.irfft(Y.div(G), 2, signal_sizes=input.shape[-2:])

    @staticmethod
    def backward(ctx, grad_output, grad_c=None):

        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            alpha, B, G, Y, wshape = ctx.intermediate_results
            channels = Y.size(2)
        elif ctx.needs_input_grad[0]:
            alpha, B, G = ctx.intermediate_results

        grad_input = grad_weights = grad_alpha = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            D = cabs(B).pow(2).unsqueeze(1)  
            T = cabs(G).pow(2).sum(dim=1).unsqueeze(0) 
            T = T.mul(alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) 
            D = D + T
            del T
            D = D.unsqueeze(-1) 

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            Z = torch.rfft(grad_output, 2) 

        if ctx.needs_input_grad[0]:
            grad_input = torch.irfft(cmul(B.unsqueeze(1), Z).div(D), 2, \
                                  signal_sizes=grad_output.shape[-2:])
            grad_input = grad_input.sum(dim=1)

        if 'B' in locals(): del B
        if ctx.needs_input_grad[2]:
            ws = tuple(int(i) for i in -(np.asarray(wshape[-2:]) // 2))
            ws = (0, 0, 0, 0) + ws
            U = cmul(conj(Z), Y.div(D.pow(2))) 
            U = U[..., 0].unsqueeze(-1).unsqueeze(2) 
            U = U.mul(G.unsqueeze(0))  
            U = torch.irfft(U, 2, signal_sizes=grad_output.shape[-2:])  
            U = utils.shift_transpose(U, ws, bc='circular')
            U = U[..., 0:wshape[3], 0:wshape[4]]  
            grad_weights = -2 * U.mul(alpha.unsqueeze(0).unsqueeze(2).unsqueeze(-1).unsqueeze(-1))
            del U
            grad_weights = grad_weights.sum(dim=0)
            if wshape[2] == 1:
                grad_weights = grad_weights.sum(dim=2, keepdim=True)
            if wshape[0] == 1 and alpha.size(0) != 1:
                grad_weights = grad_weights.sum(dim=0)

        if 'Z' in locals(): del Z
        if ctx.needs_input_grad[3]:
            Y = Y.mul(cabs(G).pow(2).sum(dim=1).unsqueeze(0).unsqueeze(-1)) 
            Y = Y.div(D.pow(2))
            Y = torch.irfft(Y, 2, signal_sizes=grad_output.shape[-2:])
            Y = Y.mul(-alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            Y = Y.mul(grad_output)
            grad_alpha = Y.sum(dim=4).sum(dim=3).sum(dim=0)
            if channels != 1 and alpha.size(-1) == 1:
                grad_alpha = grad_alpha.sum(dim=-1, keepdim=True)

        return grad_input, None, grad_weights, grad_alpha


class WienerUNet(torch.nn.Module):

    def __init__(self):
        '''
        Deconvolution function for a batch of images. Although the regularization
        term does not have a shape of Tikhonov regularizer, with a slight abuse of notations
        the function is called WienerUNet.

        The function is built upon the iterative gradient descent scheme:

        x_k+1 = x_k - lamb[K^T(Kx_k - y) + exp(alpha)*reg(x_k)]

        Initial parameters are:
        regularizer: a neural network to parametrize the prior on each iteration x_k.
        alpha: power of the trade-off coefficient 
        lamb: step of the gradient descent algorithm
        '''
        super(WienerUNet, self).__init__()
        self.regularizer = UNet(mode='instance')
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]))
        self.lamb = nn.Parameter(torch.FloatTensor([0.3]))

    def forward(self, x, y, ker):
        '''
        Function that performs one iteration of the gradient descent scheme of the deconvolution algorithm.

        x: (torch.(cuda.)Tensor) Image, restored with the previous iteration of the gradient descent scheme, B x C x H x W
        y: (torch.(cuda.)Tensor) Input blurred and noisy image B x C x H x W
        ker: (torch.(cuda.)Tensor) Tensor of PSFs B x C x Hk x Wk
         
        (self.) alpha: (torch.(cuda.)Tensor) Power of a trade-off coefficient exp(alpha) 
        (self.) lamb: (torch.(cuda.)Tensor) Gradient descent step
        returns: (torch.(cuda.)Tensor) Restored image B x C x Hk x Wk
        '''

        x_filtered = utils.imfilter2D_SpatialDomain(x, ker, padType='symmetric', mode="conv")
        Kx_y = x_filtered - y

        y_filtered = utils.imfilter_transpose2D_SpatialDomain(Kx_y, ker, padType='symmetric', mode="conv")

        regul = torch.exp(self.alpha) * self.regularizer(x)

        brackets = y_filtered + regul
        out = x - self.lamb * brackets

        return out
   