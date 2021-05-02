import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from collections import OrderedDict
from dataloader import CellDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import *
from instruments import model_load
from networks import UNet
from functions import *
import os
from ssim import ssim
from skimage import io
from utils import *

import argparse

parser = argparse.ArgumentParser(description='Image deconvolution with trainable algorithms')

parser.add_argument('--method', type=str, help='Define algorithm to run')
parser.add_argument('--model_path', type=str, help='Path to the models')
parser.add_argument('--test_std', type=float, help='Standard deviation of the Gaussian noise')
parser.add_argument('--visual', type=int, help='Save (visual=1) or not (visual=0) evaluated results during test')
parser.add_argument('--use_gpu', type=int, help='Use GPU (use_gpu=1) or CPU (use_gpu=0) for evaluation')


##############################################____ALGORITHMS____#################################################

def test_unet(root,\
              psf_path,
              method,\
              std, \
              model_path,\
              visual, \
              use_gpu,\
              b_size=1):
    
    """
    Model UNet
    """
    
    model_name = method + '_gaussian'
    save_images_path = './Results/' + model_name + '_std_' + str(std).replace('.', '') + '/'

    test_dataset = CellDataset(root, psf_path, 'gaussian', 1.0, std)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False, num_workers=1)

    model = UNet(mode='batch')
    
    state_dict = torch.load(os.path.join(model_path, model_name))
    state_dict = state_dict['model_state_dict'] 
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    
    if use_gpu==1:
        model.cuda()

    psnr_values_test = []
    ssim_values_test = []

    distorted_psnr_test = []
    distorted_ssim_test = []

    for i_batch, ((gt, image), psf, index, image_name, _, std) in enumerate(tqdm(test_loader)):
        
        image = image.reshape((b_size, 1, image.shape[-2], image.shape[-1]))

        gt = gt.reshape((b_size, 1, gt.shape[-2],gt.shape[-1]))
        
        if use_gpu==1:
            image = image.cuda()
            gt = gt.cuda()
        
        distorted_psnr = calc_psnr(image.clamp(gt.min(), gt.max()), gt)
        distorted_ssim = ssim(image.clamp(gt.min(), gt.max()), gt)
        
        output = model(image)

        psnr_test = calc_psnr(output.clamp(gt.min(), gt.max()), gt)
        s_sim_test = ssim(output.clamp(gt.min(), gt.max()), gt)

        psnr_values_test.append(psnr_test.item())
        ssim_values_test.append(s_sim_test.item())

        distorted_psnr_test.append(distorted_psnr.item())
        distorted_ssim_test.append(distorted_ssim.item())
             
        #Save image
        if visual==1:
            
            if not os.path.exists(save_images_path):
                    os.makedirs(save_images_path, exist_ok=True)
                    
            io.imsave(os.path.join(save_images_path, 'output_' + str(image_name[0][:-4]) + '_' + \
                      str(model_name) + '_' + str(std.item()).replace('.', '') + '.png'), np.uint8(output[0][0].detach().cpu().numpy().clip(0,1) * 255.))

    print('Test on Gaussian noise with %.3f std: PSNR %.2f, SSIM %.4f, distorted PSNR %.2f, distorted SSIM %.4f' % (std, np.array(psnr_values_test).mean(), \
                                                                                              np.array(ssim_values_test).mean(), \
                                                                                              np.array(distorted_psnr_test).mean(), \
                                                                                              np.array(distorted_ssim_test).mean()))

    return

def test_wiener(root,\
                psf_path,
                method,\
                model_path,\
                std,\
                visual,\
                use_gpu,\
                b_size=1):
    
    '''
    Models: Wiener filter + common kernels
            Wiener filter + per-image kernels
    '''
    
    model_name = method + '_gaussian'
    save_images_path = './Results/' + model_name + '_std_' + str(std).replace('.', '') + '/'

    test_dataset = CellDataset(root, psf_path, 'gaussian', 1.0, std)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

    model = model_load('gaussian', method, model_path)
    model.eval()
    
    if use_gpu==1:
        model.cuda()

    psnr_values_test = []
    ssim_values_test = []

    distorted_psnr_test = []
    distorted_ssim_test = []

    for i_batch, ((gt, image), psf, index, image_name, _, std) in enumerate(tqdm(test_loader)):
        image = image.reshape((b_size, 1, image.shape[-2], image.shape[-1]))

        gt = gt.reshape((b_size, 1, gt.shape[-2], gt.shape[-1]))

        psf = psf.reshape([b_size, 1, psf.shape[-2], psf.shape[-1]]).float()
        
        if use_gpu==1:
            image = image.cuda()
            gt = gt.cuda()
            psf = psf.cuda()
        
        distorted_psnr = calc_psnr(image.clamp(gt.min(), gt.max()), gt)
        distorted_ssim = ssim(image.clamp(gt.min(), gt.max()), gt)

        image = EdgeTaper.apply(pad_psf_shape(image, psf), psf[0][0])

        out = model((image, psf))

        out = crop_psf_shape(out, psf)

        psnr_test = calc_psnr(out.clamp(gt.min(), gt.max()), gt)
        s_sim_test = ssim(out.clamp(gt.min(), gt.max()), gt)

        psnr_values_test.append(psnr_test.item())
        ssim_values_test.append(s_sim_test.item())
        
        distorted_psnr_test.append(distorted_psnr.item())
        distorted_ssim_test.append(distorted_ssim.item())
            
         #Save image
        if visual==1:
            
            if not os.path.exists(save_images_path):
                    os.makedirs(save_images_path, exist_ok=True)
                    
            io.imsave(os.path.join(save_images_path, 'output_' + str(image_name[0][:-4]) + '_' + \
                      str(model_name) + '_' + str(std.item()).replace('.', '') + '.png'), np.uint8(out[0][0].detach().cpu().numpy().clip(0,1) * 255.))

    print('Tested on Gaussian noise with %.3f std: PSNR %.2f, SSIM %.4f, distorted PSNR %.2f, distorted SSIM %.4f' % (std, np.array(psnr_values_test).mean(), \
                                                                                              np.array(ssim_values_test).mean(), \
                                                                                              np.array(distorted_psnr_test).mean(), \
                                                                                              np.array(distorted_ssim_test).mean()))

    return

def test_wienerUnet(data_path,\
                    psf_path,
                    method,\
                    model_path,\
                    std, \
                    visual, \
                    use_gpu,\
                    b_size=1):
    
    '''
    Gradient descent scheme + predicted with UNet per-image gradient of a regularizer
    '''
    
    model_name = method + '_gaussian'
    save_images_path = './Results/' + model_name + '_std_' + str(std).replace('.', '') + '/'


    test_dataset = CellDataset(data_path, psf_path, 'gaussian', 1.0, std)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

    model = model_load('gaussian', method, model_path)
    model.eval()
    
    if use_gpu==1:
        model.cuda()

    psnr_values_test = []
    ssim_values_test = []

    distorted_psnr_test = []
    distorted_ssim_test = []

    for i_batch, ((gt, image), psf, index, image_name, _, std) in enumerate(tqdm(test_loader)):
        image = image.reshape((b_size, 1, image.shape[-2], image.shape[-1]))

        gt = gt.reshape((b_size, 1, gt.shape[-2], gt.shape[-1]))

        psf = psf.reshape([b_size, 1, psf.shape[-2], psf.shape[-1]]).float()
        
        if use_gpu==1:
            image = image.cuda()
            gt = gt.cuda()
            psf = psf.cuda()
        
        distorted_psnr = calc_psnr(image.clamp(gt.min(), gt.max()), gt)
        distorted_ssim = ssim(image.clamp(gt.min(), gt.max()), gt)

        image = EdgeTaper.apply(pad_psf_shape(image, psf), psf[0][0])

        out = model(image, psf)

        out = crop_psf_shape(out, psf)

        psnr_test = calc_psnr(out.clamp(gt.min(), gt.max()), gt)
        s_sim_test = ssim(out.clamp(gt.min(), gt.max()), gt)

        psnr_values_test.append(psnr_test.item())
        ssim_values_test.append(s_sim_test.item())
        
        distorted_psnr_test.append(distorted_psnr.item())
        distorted_ssim_test.append(distorted_ssim.item())
            
         #Save image
        if visual==1:
            
            if not os.path.exists(save_images_path):
                    os.makedirs(save_images_path, exist_ok=True)
                    
            io.imsave(os.path.join(save_images_path, 'output_' + str(image_name[0][:-4]) + '_' + \
                      str(model_name) + '_' + str(std.item()).replace('.', '') + '.png'), np.uint8(out[0][0].detach().cpu().numpy().clip(0,1) * 255.))

    print('Tested on Gaussian noise with %.3f std: PSNR %.2f, SSIM %.4f, distorted PSNR %.2f, distorted SSIM %.4f' % (std,\
                                                                                                             np.array(psnr_values_test).mean(), \
                                                                                              np.array(ssim_values_test).mean(), \
                                                                                              np.array(distorted_psnr_test).mean(), \
                                                                                              np.array(distorted_ssim_test).mean()))

    return
###############################################################################################
import numpy as np
def vnorm_b(x):
    r"""returns the 2-norm of a vector.
    """
    assert torch.is_tensor(x), "Only tensor inputs are supported"
    b = x.shape[0]
    return x.reshape(b, -1).norm(p=2, dim=1)


def vdot_b(x, y):
    """ returns the vector product of x, y."""
    assert torch.is_tensor(x) and torch.is_tensor(y), "Only tensor inputs are supported."
    b = x.shape[0]
    return (x.reshape(b, -1) * y.reshape(b, -1)).sum(dim=1)


def cg_solver_b(A, filters_out, reg_weight, b, x=None, M=None, maxiter=50, tol=1e-1, restart=50, info=None):
    r"""Conjugate Gradient method for solving the problem Ax = b.
    Parameters
    ----------
    A : function.
        Function that performs the linear operation A(x).
    b : ndarray.
        The solution of the linear system Ax = b.
    x : ndarray.
        Initial solution of the system. (Default: None)
    M : function.
        Function that performs precondtioning M(x).

    maxiter : int, optional.
              Maximum number of iterations. (Default: 100)
    tol : float, optional.
          Stopping threshold. (Default: 1e-1)
    restart : int, optional.
              Indicates the frequency of computing the residual as
              r[k] = b - A(x[k]) instead of the recursion formula
              r[k] = r[k-1] - alpha*A(r[k-1]). (Default: 50)
    Returns
    -------
    x : ndarray
        The estimate of the solution of the linear system
    iter : int
        The number of CG iterations that run.
    alpha : float
        Last the computed step size alpha
    info : dict
        dictionary with iteration information
    """
    if x is None:
        x = torch.zeros_like(b)
    r = b - A(x, filters_out, reg_weight)  # residual
    if M is None:
        d = r
        delta = vnorm_b(r) ** 2
    else:
        d = M(r)
        delta = vdot_b(r, d)
    threshold = tol * delta #tol * tol * delta

    # prepare info

    info = {'res': torch.zeros(maxiter, x.shape[0], dtype=x.dtype, device=x.device),
            'inires': torch.sqrt(delta),
            'converged': torch.zeros(x.shape[0], device=x.device).bool(),
            'cg_iter': -1}

    converged_mask = torch.zeros(x.shape[0], device=x.device).bool()
    if (delta <= threshold).any():
        converged_mask = delta <= threshold
        info['converged'][converged_mask] = True
    cg_iter = maxiter

    for i in range(maxiter):

        q = A(d, filters_out, reg_weight)

        alpha = delta / vdot_b(d, q)
        alpha[converged_mask] = 0

        x = x + alpha[:, None] * d * (~converged_mask[:, None])
        # x = x + alpha[:, None] * d * (float(~converged_mask[:, None]))

        if ((i + 1) % restart) == 0:
            r = b - A(x, filters_out, reg_weight)
        else:
            r = r - alpha[:, None] * q

        delta_old = delta

        # delta_old[converged_mask] = delta_old[converged_mask].clamp(min=1e-16)
        # delta_old[converged_mask].clamp_(min=1e-16)
        if M is None:
            delta = vnorm_b(r) ** 2
        else:
            s = M(r)
            delta = vdot_b(r, s)

        info['res'][i, ~converged_mask] = torch.sqrt(delta[~converged_mask])
        if (delta < threshold).any():
            cg_iter = i + 1

            converged_mask = converged_mask | (delta <= threshold)
            info['converged'][converged_mask] = True

            if info['converged'].all():
                break

        if M is None:
            d = r + (delta / delta_old)[:, None] * d
        else:
            d = s + (delta / delta_old)[:, None] * d

    info['cg_iter'] = cg_iter
    info['res'] = info['res'][:cg_iter]
    return x, info

class ConugateGradient_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_f, filters_out, reg_weight, b, x0=None, M=None, maxiter=50, tol=1e-2, restart=50):
        # ctx is a context object that can be used to stash information
        # for backward computation
        x, cg_info = cg_solver_b(A_f, filters_out, reg_weight, b, x=x0, M=M, maxiter=maxiter, tol=tol, restart=restart)
        info = cg_info
        ctx.info = cg_info
        ctx.A_f = A_f
        ctx.filters_out = filters_out
        ctx.reg_weight = reg_weight
        ctx.b = b
        ctx.x = x.detach()
        ctx.x0 = x0
        ctx.M = M
        ctx.maxiter = maxiter
        ctx.tol = tol
        ctx.restart = restart
        return x, cg_info['res'], cg_info['converged']

    @staticmethod
    def backward(ctx, grad_output, nonused1, nonused2):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        info = None
        if grad_output.sum() == 0:
            grad_over_b = torch.zeros_like(grad_output)
        else:
            with torch.no_grad():
                grad_over_b, cg_info = cg_solver_b(ctx.A_f, ctx.filters_out, ctx.reg_weight, \
                                                   grad_output, x=None, M=ctx.M, maxiter=ctx.maxiter, tol=ctx.tol, restart=ctx.restart)

        with torch.enable_grad():
            foo_inp = ctx.x.clone().detach()
            foo_inp.requires_grad = True
            filters_out = ctx.filters_out.clone().detach().requires_grad_(True)
            reg_weight = ctx.reg_weight.clone().detach().requires_grad_(True)
            foo_inp = ctx.A_f(foo_inp, filters_out, reg_weight, create_graph=True, retain_graph=True)
            grad_over_ = torch.autograd.grad(foo_inp, [filters_out, reg_weight], grad_outputs=-grad_over_b,
                                             create_graph=True, retain_graph=True, allow_unused=False)
            grad_over_fo = grad_over_[0]
            grad_over_rw = grad_over_[1]

            del foo_inp
        if ctx.x0 is not None:
            grad_over_x0 = torch.zeros_like(grad_output)
        else:
            grad_over_x0 = None
        del ctx.b, ctx.x0, ctx.A_f, ctx.M, nonused1, nonused2, ctx.x, ctx.maxiter, ctx.tol, ctx.restart, ctx.info
        torch.cuda.empty_cache()
        return None, grad_over_fo, grad_over_rw, grad_over_b, grad_over_x0, None, None, None, None

class Im2Col(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, padding):
        ctx.shape, ctx.kernel_size, ctx.padding = (x.shape[2:], kernel_size, padding)
        return F.unfold(x, kernel_size=kernel_size, padding=padding)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            shape, ks, padding = ctx.shape, ctx.kernel_size, ctx.padding
            return (
                F.fold(grad_output, shape, kernel_size=ks, padding=padding),
                None,
                None,
            )


def im2col(x, kernel_size, padding):
    return Im2Col.apply(x, kernel_size, padding)

def spatial_conv(x, kernels):
    # calculates spatially varying convolution for a batch
    batch, C, H, W = x.shape
    kern_supp = kernels.shape[3]
    padding = kern_supp // 2
    kernels = kernels.reshape(batch, 9, -1).transpose(1, 2) #TODO: check shape here again, maybe this causes the artifacts
    # kernels = kernels.reshape(batch, kernels.shape[1], -1).transpose(1, 2)
    out = []
    for c in range(x.shape[1]):
        inp_unf = im2col(x[:, c][:, None], (kern_supp, kern_supp), padding=padding)
        b, c_, pix = inp_unf.shape
        inp_unf = inp_unf.transpose(1, 2).reshape(b * pix, c_)
        kernels = kernels.reshape(b * pix, c_)
        filtered_inp_unf = inp_unf[:, None].bmm(kernels[..., None])
        filtered_inp_unf = filtered_inp_unf.reshape(b, pix)
        filtered_inp_unf = filtered_inp_unf.reshape(batch, C, int(np.sqrt(pix)), int(np.sqrt(pix)))
        out.append(filtered_inp_unf)

    out = torch.cat(out, dim=1)
    return out    
    
class Wiener_KPN_SA(nn.Module):
    def __init__(self, maxiter=300, tol=1e-6, restart=50):
        super(Wiener_KPN_SA, self).__init__()
        self.maxiter = maxiter
        self.tol = tol
        self.restart = restart
        self.cg_solver = ConugateGradient_Function
        self.info = None

        self.reg_weight = nn.Parameter(torch.Tensor([0.]))#.double())
        self.model = UNet(mode='none', n_channels=1, n_classes=9)

    def forward(self, y, k, padding=1, x0=None, M_f=None):
        B, C, H, W = y.shape
        # predict kernels
        filters_out = self.model(y)#.double()
        # this is the fix, every variable needs a different name, else grad is None
        # and as such no progress during training
        filters_out_ = filters_out.reshape(B, C, -1, H, W)
        filters_out__ = F.normalize(filters_out_, dim=2, p=1)
        filters_out___ = filters_out__.reshape(B, C, C, 3, 3, H, W)

        b = utils.imfilter_transpose2D_SpatialDomain(y, k, padType='symmetric', mode="conv").reshape(B, -1)#.double()
        b = b.requires_grad_(True)

        # k = k.double()
        # y = y.double()

    # func = 0.5||y - Kx||2_2 + 0.5exp(a)||Gx||2_2
        @torch.enable_grad()
        def func(x, k, filters_out, reg_weight, y):  
            x = x.reshape(B, C, H, W)
            k_x = utils.imfilter2D_SpatialDomain(x, k, padType='symmetric', mode="conv")
            k_x = k_x.reshape(B, -1)
            l_x = spatial_conv(x, filters_out).reshape(B, -1)
            return 0.5 * torch.norm(y.reshape(B, -1) - k_x, dim=1, p=2) ** 2 + 0.5 * torch.exp(reg_weight) * torch.norm(l_x, dim=1, p=2) ** 2

        def grad_func(x, filters_out, reg_weight, create_graph=False, retain_graph=False):
            # x = x.double()
            x = x.requires_grad_(True)
            out = func(x, k, filters_out, reg_weight, y)
            grad = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out), create_graph=create_graph, retain_graph=retain_graph)[0] #(KTK + lambda GTG)x - KTy
            k_y = utils.imfilter_transpose2D_SpatialDomain(y, k, padType='symmetric', mode="conv")
            return grad + k_y.reshape(B, -1)

        x, res, converged = self.cg_solver.apply(grad_func, filters_out___, self.reg_weight, b, x0, M_f, self.maxiter, self.tol, self.restart)
        info = {'res': res, 'converged': converged}

        # print('Conv: ', info['converged'])
        return x.reshape(B, C, H, W).float()
    
def test_wienerKPN_SA(data_path,\
                      psf_path,
                      method,\
                      model_path,\
                      std, \
                      visual, \
                      use_gpu,\
                      b_size=1):
    
    '''
    Wiener filter + predictable per-pixel kernels for each image
    '''
    
    model_name = method + '_gaussian'
    save_images_path = './Results/' + model_name + '_std_' + str(std).replace('.', '') + '/'


    test_dataset = CellDataset(data_path, psf_path, 'gaussian', 1.0, std)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

    model = model_load('gaussian', method, model_path)
    model.eval()
    
#     model = Wiener_KPN_SA()

#     state_dict = torch.load(os.path.join(model_path, model_name))
#     state_dict = state_dict['model_state_dict']
#     # create new OrderedDict that does not contain `module.`
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:]  # remove `module.`
#         new_state_dict[name] = v
#     # load params
#     model.load_state_dict(new_state_dict)
#     model.eval()

#     model = nn.DataParallel(model).cuda()

#     loss_values_test = []
#     psnr_values_test = []
#     ssim_values_test = []

#     ground_truths = []
#     restored_imgs = []
#     image_names = []

#     baseline_psnr_test = []
#     baseline_ssim_test = []

#     # model.train(False)
#     with torch.no_grad():
#         for i_batch, ((gt_batch_test, image_batch_test), psf_test, index_test, image_name_test, peak, std) in enumerate(test_loader):
#             image_batch_test = image_batch_test.reshape((b_size, 1, image_batch_test.shape[-2],
#                                                        image_batch_test.shape[-1])).cuda()

#             gt_batch_test = gt_batch_test.reshape((b_size, 1, gt_batch_test.shape[-2],
#                                                  gt_batch_test.shape[-1])).cuda()

#             psf_test = psf_test.reshape([b_size, 1, psf_test.shape[-2], psf_test.shape[-1]]).float().cuda()

#             print(psf_test.norm())

#             baseline_psnr = calc_psnr(image_batch_test.clamp(0, 1), gt_batch_test)
#             baseline_ssim = ssim(image_batch_test.clamp(0, 1), gt_batch_test)

#             image_batch_test = EdgeTaper.apply(pad_psf_shape(image_batch_test, psf_test),
#                                                   psf_test[0][0])

#             # print(psf_test.shape)
#             out_test = model(image_batch_test, psf_test)


#             out_test = crop_psf_shape(out_test, psf_test)

#             loss_test = nn.functional.l1_loss(out_test.clamp(0,1), gt_batch_test) + \
#                         nn.functional.l1_loss(get_image_grad(out_test.clamp(0,1))[0], get_image_grad(gt_batch_test)[0]) + \
#                         nn.functional.l1_loss(get_image_grad(out_test.clamp(0,1))[1], get_image_grad(gt_batch_test)[1])


#             psnr_test = calc_psnr(out_test.clamp(0,1), gt_batch_test)
#             s_sim_test = ssim(out_test.clamp(0,1), gt_batch_test)

#             loss_values_test.append(loss_test.item())
#             psnr_values_test.append(psnr_test.item())
#             ssim_values_test.append(s_sim_test.item())
#             baseline_psnr_test.append(baseline_psnr.item())
#             baseline_ssim_test.append(baseline_ssim.item())

#             print('Test: {}, {}, loss {}, PSNR {}, SSIM {}, baseline {}'.format(image_name_test[0],\
#                                                                                 index_test.item(),\
#                                                                                 loss_test.item(),
#                                                                                 psnr_test.item(),
#                                                                                 s_sim_test.item(),
#                                                                                 baseline_psnr.item()))

#             ground_truths.append(gt_batch_test[0][0].detach().cpu().numpy())
#             restored_imgs.append(out_test[0][0].detach().cpu().numpy())
#             image_names.append(image_name_test)

#             if visual==1:

#                 if not os.path.exists(save_images_path):
#                         os.makedirs(save_images_path, exist_ok=True)
#                 io.imsave(os.path.join(save_images_path, 'output_' + str(image_name_test[0][:-4]) + '_' + \
#                           str(model_name) + '_' + str(std.item()).replace('.', '') + '.png'), np.uint8(out_test[0][0].detach().cpu().numpy().clip(0,1) * 255.))



#     return 

    if use_gpu==1:
        model.cuda()
       
    psnr_values_test = []
    ssim_values_test = []

    distorted_psnr_test = []
    distorted_ssim_test = []
    
    with torch.no_grad():
        for i_batch, ((gt, image), psf, index, image_name, _, std) in enumerate(tqdm(test_loader)):
            image = image.reshape((b_size, 1, image.shape[-2], image.shape[-1]))

            gt = gt.reshape((b_size, 1, gt.shape[-2], gt.shape[-1]))

            psf = psf.reshape([b_size, 1, psf.shape[-2], psf.shape[-1]]).float()

            if use_gpu==1:
                image = image.cuda()
                gt = gt.cuda()
                psf = psf.cuda()

            distorted_psnr = calc_psnr(image.clamp(gt.min(), gt.max()), gt)
            distorted_ssim = ssim(image.clamp(gt.min(), gt.max()), gt)

            image = EdgeTaper.apply(pad_psf_shape(image, psf), psf[0][0])
            out = model(image, psf)
#             out = image

            out = crop_psf_shape(out, psf)

            psnr_test = calc_psnr(out.clamp(gt.min(), gt.max()), gt)
            s_sim_test = ssim(out.clamp(gt.min(), gt.max()), gt)

            psnr_values_test.append(psnr_test.item())
            ssim_values_test.append(s_sim_test.item())

            distorted_psnr_test.append(distorted_psnr.item())
            distorted_ssim_test.append(distorted_ssim.item())

             #Save image
            if visual==1:

                if not os.path.exists(save_images_path):
                        os.makedirs(save_images_path, exist_ok=True)

                io.imsave(os.path.join(save_images_path, 'output_' + str(image_name[0][:-4]) + '_' + \
                          str(model_name) + '_' + str(std.item()).replace('.', '') + '.png'), np.uint8(out[0][0].detach().cpu().numpy().clip(0,1) * 255.))


    print('Tested on Gaussian noise with %.3f std: PSNR %.2f, SSIM %.4f, distorted PSNR %.2f, distorted SSIM %.4f' % (std,\
                                                                                                             np.array(psnr_values_test).mean(), \
                                                                                              np.array(ssim_values_test).mean(), \
                                                                                              np.array(distorted_psnr_test).mean(), \
                                                                                              np.array(distorted_ssim_test).mean()))

    return


###############################################################################################################################

if __name__ == '__main__':
    opt = parser.parse_args()

    print('========= Selected model and inference parameters =============')
    print(opt)
    print('===========================================================================')
    print('\n')

    data_path = './dataset/'
    psf_path  = './dataset/'

    if opt.method=='WFK':
        test_wiener(data_path, psf_path, opt.method, opt.model_path, opt.test_std, opt.visual, opt.use_gpu)
    if opt.method=='WF_KPN':
        test_wiener(data_path, psf_path, opt.method, opt.model_path, opt.test_std, opt.visual, opt.use_gpu)
    if opt.method == 'WF_KPN_SA':
        test_wienerKPN_SA(data_path, psf_path, opt.method, opt.model_path, opt.test_std, opt.visual, opt.use_gpu)
    if opt.method == 'WF_UNet':
        test_wienerUnet(data_path, psf_path, opt.method, opt.model_path, opt.test_std, opt.visual, opt.use_gpu)
    if opt.method == 'UNet':
            test_unet(data_path, psf_path, opt.method, opt.test_std, opt.model_path, opt.visual, opt.use_gpu)
