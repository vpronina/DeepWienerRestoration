import torch
import torch.nn as nn
from functions import Wiener, WienerUNet
import torch.nn.functional as F
from networks import KPN, UNet
import utils


class WienerFilter(nn.Module):
    
    '''
    Module that applies Wiener deconvolution algorithm with 
    identical learnable regularization kernels
    '''

    def __init__(self, n_filters, filter_size):
        
        '''
        n_filters (int): Number of learnable filters in a group
        filter_size (int): Size of each filter in the group
        '''
        super(WienerFilter, self).__init__()

        self.alpha = torch.FloatTensor([0.0])
        self.alpha = self.alpha.unsqueeze(-1)
        self.alpha = nn.Parameter(self.alpha)

        self.g_ker = torch.Tensor(n_filters, 1, filter_size, filter_size)
        self.g_ker = nn.Parameter(self.g_ker)
        utils.dct(self.g_ker)

    def forward(self, inputs):
        '''
        inputs: (torch.(cuda.)Tensor) Model inputs: tensor of input image B x C x H x W and 
        tensor of PSFs B x C x Hk x Wk
        
        (self.) alpha: (torch.(cuda.)Tensor) Power of a trade-off coefficient exp(alpha) 1 x 1
        (self.) g_ker: (torch.(cuda.)Tensor) Group of identical learnable regularization kernels D x 1 x Hg x Wg
        
        returns: (torch.(cuda.)Tensor) Output of the Wiener-Kolmogorov filter with learnable 
        regularization kernels B x C x H x W
        '''
        
        y, ker = inputs
        nf, nc, nx, ny = self.g_ker.shape
        g_ker = self.g_ker

        g_ker = F.normalize(g_ker.reshape(g_ker.shape[0], -1), dim=1, p=1)
        g_ker = g_ker.reshape(nf, nc, nx, ny)

        output = Wiener().apply(y, ker, g_ker, self.alpha)
        output = output.squeeze(1)
        return output

class WienerFilterKPN(nn.Module):
    '''
        Module that uses KPN to predict individual kernel for each input image and then
        applies Wiener deconvolution algorithm with predicted regularization kernels per-image
    '''

    def __init__(self, n_filters, filter_size):
        '''
         n_filters (int): Number of predictable filters in a group
         filter_size (int): Size of each filter in the group
        '''
        super(WienerFilterKPN, self).__init__()

        self.alpha = torch.FloatTensor([0.0])
        self.alpha = self.alpha.unsqueeze(-1)
        self.alpha = nn.Parameter(self.alpha)

        self.regularizer = KPN(mode='instance', N=n_filters, K=filter_size)
        self.function = Wiener()

    def forward(self, inputs):
        '''
        inputs: (torch.(cuda.)Tensor) Model inputs: tensor of input images B x C x H x W and 
        tensor of PSFs B x C x Hk x Wk
        
        (self.) alpha: (torch.(cuda.)Tensor) Power of a trade-off coefficient exp(alpha) 1 x 1
        (self.) g_ker: (torch.(cuda.)Tensor) Group of identical predictable regularization kernels D x 1 x Hg x Wg
        
        returns: (torch.(cuda.)Tensor) Output of the Wiener-Kolmogorov filter with 
        predictable regularization kernels B x C x H x W
        '''   

        y, ker = inputs
        output = torch.zeros_like(y)

        res = self.regularizer(y)

        for i in range(y.shape[0]): 
            inp = y[i]            
            psf = ker[i]            
            g_ker = res[i]

            nf, nc, nx, ny = g_ker.shape
            g_ker = F.normalize(g_ker.reshape(g_ker.shape[0], -1), dim=1, p=1)
            g_ker = g_ker.reshape(nf, nc, nx, ny)

            out = self.function.apply(inp[None], psf[None], g_ker, self.alpha)
            tmp = out[0].squeeze(1)
            output[i] = tmp

        return output

class WienerFilter_UNet(nn.Module):
    '''
    Module that uses UNet to predict individual gradient of a regularizer for each input image and then
    applies gradient descent scheme with predicted gradient of a regularizers per-image
    '''
    def __init__(self):

        super(WienerFilter_UNet, self).__init__()
        self.function = WienerUNet()

    def forward(self, y, ker, n_iter=10):
        '''
        y: (torch.(cuda.)Tensor) Tensor of input images of shape B x C x H x W
        ker: (torch.(cuda.)Tensor) Tensor of PSFs of shape B x C x Hk x Wk
        n_iter: (int) Number of gradient descent iterations
        returns: (torch.(cuda.)Tensor) Output of the gradient descent scheme B x C x H x W
        '''
        
        output = y.clone()

        for i in range(n_iter):
            output = self.function(output, y, ker)

        return output