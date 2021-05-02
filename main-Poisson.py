import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import OrderedDict

from dataloader import CellDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import *
from networks import UNet
from functions import *
from instruments import model_load
import os
from ssim import ssim
from skimage import io
from utils import *

import argparse
parser = argparse.ArgumentParser(description='Image deconvolution with trainable algorithms')

parser.add_argument('--method', type=str, help='Define algorithm to run')
parser.add_argument('--model_path', type=str, help='Path to the models')
parser.add_argument('--visual', type=int, help='Save (visual=1) or not (visual=0) evaluated results during test')
parser.add_argument('--use_gpu', type=int, help='Use GPU (use_gpu=1) or CPU (use_gpu=0) for evaluation')
parser.add_argument('--test_scale', type=float, help='Peak value for the ground truth image rescaling to simulate various levels of Poisson noise')

##############################################____ALGORITHMS____#################################################

def test_unet(root,\
              psf_path,
              method,\
              scale, \
              model_path,\
              visual, \
              use_gpu,
              b_size=1):

    '''
    Model UNet
    '''
    
    model_name = method + '_poisson'
    save_images_path = './Results/' + model_name + '_peak_' + str(int(scale)) + '/'

    test_dataset = CellDataset(root, psf_path, 'poisson', scale, 0.0)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False, num_workers=1)

    model = UNet(mode='batch')
    state_dict = torch.load(os.path.join(model_path, model_name))
    state_dict = state_dict['model_state_dict'] 
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict)
    
    if use_gpu==1:
        model.cuda()
        
    model.eval()

    psnr_values_test = []
    ssim_values_test = []

    distorted_psnr_test = []
    distorted_ssim_test = []

    with torch.no_grad():
        for i_batch, ((gt, image), psf, index, image_name, peak, _) in enumerate(tqdm(test_loader)):
            image = image.reshape((b_size, 1, image.shape[-2], image.shape[-1]))

            gt = gt.reshape((b_size, 1, gt.shape[-2], gt.shape[-1]))
            
            if use_gpu==1:
                image = image.cuda()
                gt = gt.cuda()
            
            for l in range(gt.shape[0]):
                image[l] = image[l] / gt[l].max()
                gt[l] /= gt[l].max()

            output = model(image)

            distorted_psnr = calc_psnr(image.clamp(0,1), gt)
            distorted_ssim = ssim(image.clamp(0,1), gt)

            psnr_test = calc_psnr(output.clamp(0,1), gt)
            s_sim_test = ssim(output.clamp(0,1), gt)

            psnr_values_test.append(psnr_test.item())
            ssim_values_test.append(s_sim_test.item())

            distorted_psnr_test.append(distorted_psnr.item())
            distorted_ssim_test.append(distorted_ssim.item())
            
            #Save image
            if visual==1:
                
                if not os.path.exists(save_images_path):
                    os.makedirs(save_images_path, exist_ok=True)
                    
                io.imsave(os.path.join(save_images_path, 'output_' + str(image_name[0][:-4]) + '_' + \
                          str(model_name) + '_' + str(int(scale)) + '.png'), np.uint8(output[0][0].detach().cpu().numpy().clip(0,1) * 255.))
                

    print('Test on Poisson noise with peak %d: PSNR %.2f, SSIM %.4f, distorted PSNR %.2f, distorted SSIM %.4f' % (peak, np.array(psnr_values_test).mean(), \
                                                                                              np.array(ssim_values_test).mean(), \
                                                                                              np.array(distorted_psnr_test).mean(), \
                                                                                              np.array(distorted_ssim_test).mean()))

    return

def test_wiener(root,\
                psf_path,
                method,\
                scale, \
                model_path,\
                visual, \
                use_gpu, \
                b_size=1):
    
    '''
    Models: Wiener filter + identical learnable kernels
            Wiener filter + predictable per-image kernels
    '''
    
    model_name = method + '_poisson'
    save_images_path = './Results/' + model_name + '_peak_' + str(int(scale)) + '/'

    test_dataset = CellDataset(root, psf_path, 'poisson', scale, 0.0)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False, num_workers=1)

    model = model = model_load('poisson', method, model_path)
    
    if use_gpu==1:
        model = model.cuda()
        
    model.eval()

    psnr_values_test = []
    ssim_values_test = []

    ground_truths = []
    restored_imgs = []
    image_names = []

    distorted_psnr_test = []
    distorted_ssim_test = []

    with torch.no_grad():
        for i_batch, ((gt, image), psf, index, image_name, peak, _) in enumerate(tqdm(test_loader)):
            
            image = image.reshape((b_size, 1, image.shape[-2], image.shape[-1]))

            gt = gt.reshape((b_size, 1, gt.shape[-2], gt.shape[-1]))

            psf = psf.reshape([b_size, 1, psf.shape[-2], psf.shape[-1]]).float()
            
            if use_gpu==1:
                image = image.cuda()
                gt = gt.cuda()
                psf = psf.cuda()

            image_batch_tmp = image.clone()

            image = anscombe(image)

            image= EdgeTaper.apply(pad_psf_shape(image, psf), psf[0][0])

            out = model((image, psf))

            out = exact_unbiased(out)

            out = crop_psf_shape(out, psf)

            for j in range(out.shape[0]):
                out[j] = out[j] / gt[j].max()
                image_batch_tmp[j] = image_batch_tmp[j] / gt[j].max()
                gt[j] /= gt[j].max()        

            distorted_psnr = calc_psnr(image_batch_tmp.clamp(0,1), gt)
            distorted_ssim = ssim(image_batch_tmp.clamp(0,1), gt)

            psnr_test = calc_psnr(out.clamp(0,1), gt)
            s_sim_test = ssim(out.clamp(0,1), gt)

            psnr_values_test.append(psnr_test.item())
            ssim_values_test.append(s_sim_test.item())
            distorted_psnr_test.append(distorted_psnr.item())
            distorted_ssim_test.append(distorted_ssim.item())
            
            #Save image
            if visual==1:
                
                if not os.path.exists(save_images_path):
                    os.makedirs(save_images_path, exist_ok=True)
                    
                io.imsave(os.path.join(save_images_path, 'output_' + str(image_name[0][:-4]) + '_' + \
                          str(model_name) + '_' + str(int(scale)) + '.png'), np.uint8(out[0][0].detach().cpu().numpy().clip(0,1) * 255.))

    print('Test on Poisson noise with peak %d: PSNR %.2f, SSIM %.4f, distorted PSNR %.2f, distorted SSIM %.4f' % (peak, np.array(psnr_values_test).mean(), \
                                                                                              np.array(ssim_values_test).mean(), \
                                                                                              np.array(distorted_psnr_test).mean(), \
                                                                                              np.array(distorted_ssim_test).mean()))
    return

def test_wienerUnet(data_path,\
                    psf_path,
                    method,\
                    scale, \
                    model_path,\
                    visual, \
                    use_gpu,\
                    n_iter=10,\
                    b_size=1):
    '''
    Gradient descent scheme + predicted with UNet per-image gradient of a regularizer
    '''
    
    model_name = method + '_poisson'
    save_images_path = './Results/' + model_name + '_peak_' + str(int(scale)) + '/'

    test_dataset = CellDataset(data_path, psf_path, 'poisson', scale, 0.0)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

    model = model_load('poisson', method, model_path)
    model.eval()
    
    if use_gpu==1:
        model.cuda()

    psnr_values_test = []
    ssim_values_test = []

    distorted_psnr_test = []
    distorted_ssim_test = []

    with torch.no_grad():
    
        for i_batch, ((gt, image), psf, index, image_name, peak, _) in enumerate(tqdm(test_loader)):
            
            image = image.reshape((b_size, 1, image.shape[-2], image.shape[-1]))

            gt = gt.reshape((b_size, 1, gt.shape[-2], gt.shape[-1]))

            psf = psf.reshape([b_size, 1, psf.shape[-2], psf.shape[-1]]).float()
            
            if use_gpu==1:
                image = image.cuda()
                gt = gt.cuda()
                psf = psf.cuda()

            image_batch_tmp = image.clone()

            image = anscombe(image)

            image= EdgeTaper.apply(pad_psf_shape(image, psf), psf[0][0])

            out = model(image, psf, n_iter)

            out = exact_unbiased(out)

            out = crop_psf_shape(out, psf)

            for j in range(out.shape[0]):
                out[j] = out[j] / gt[j].max()
                image_batch_tmp[j] = image_batch_tmp[j] / gt[j].max()
                gt[j] /= gt[j].max()        

            distorted_psnr = calc_psnr(image_batch_tmp.clamp(0,1), gt)
            distorted_ssim = ssim(image_batch_tmp.clamp(0,1), gt)

            psnr_test = calc_psnr(out.clamp(0,1), gt)
            s_sim_test = ssim(out.clamp(0,1), gt)

            psnr_values_test.append(psnr_test.item())
            ssim_values_test.append(s_sim_test.item())
            distorted_psnr_test.append(distorted_psnr.item())
            distorted_ssim_test.append(distorted_ssim.item())
            
            #Save image
            if visual==1:
                
                if not os.path.exists(save_images_path):
                    os.makedirs(save_images_path, exist_ok=True)
                    
                io.imsave(os.path.join(save_images_path, 'output_' + str(image_name[0][:-4]) + '_' + \
                          str(model_name) + '_' + str(int(scale)) + '.png'), np.uint8(out[0][0].detach().cpu().numpy().clip(0,1) * 255.))

    print('Test on Poisson noise with peak %d: PSNR %.2f, SSIM %.4f, distorted PSNR %.2f, distorted SSIM %.4f' % (peak, np.array(psnr_values_test).mean(), \
                                                                                              np.array(ssim_values_test).mean(), \
                                                                                              np.array(distorted_psnr_test).mean(), \
                                                                                              np.array(distorted_ssim_test).mean()))

    return

def test_wienerKPN_SA(data_path,\
                      psf_path,
                      method,\
                      scale, \
                      model_path,\
                      visual, \
                      use_gpu,\
                      b_size=1):
    '''
    Wiener filter + predictable per-pixel kernels for each image
    '''
    
    model_name = method + '_poisson'
    save_images_path = './Results/' + model_name + '_peak_' + str(int(scale)) + '/'

    test_dataset = CellDataset(data_path, psf_path, 'poisson', scale, 0.0)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

    model = model_load('poisson', method, model_path)
    model.eval()
    
    if use_gpu==1:
        model.cuda()

    psnr_values_test = []
    ssim_values_test = []

    distorted_psnr_test = []
    distorted_ssim_test = []

    with torch.no_grad():
    
        for i_batch, ((gt, image), psf, index, image_name, peak, _) in enumerate(tqdm(test_loader)):
            
            image = image.reshape((b_size, 1, image.shape[-2], image.shape[-1]))

            gt = gt.reshape((b_size, 1, gt.shape[-2], gt.shape[-1]))

            psf = psf.reshape([b_size, 1, psf.shape[-2], psf.shape[-1]]).float()
            
            if use_gpu==1:
                image = image.cuda()
                gt = gt.cuda()
                psf = psf.cuda()

            image_batch_tmp = image.clone()

            image = anscombe(image)

            image= EdgeTaper.apply(pad_psf_shape(image, psf), psf[0][0])

            out = model(image, psf)

            out = exact_unbiased(out)

            out = crop_psf_shape(out, psf)

            for j in range(out.shape[0]):
                out[j] = out[j] / gt[j].max()
                image_batch_tmp[j] = image_batch_tmp[j] / gt[j].max()
                gt[j] /= gt[j].max()        

            distorted_psnr = calc_psnr(image_batch_tmp.clamp(0,1), gt)
            distorted_ssim = ssim(image_batch_tmp.clamp(0,1), gt)

            psnr_test = calc_psnr(out.clamp(0,1), gt)
            s_sim_test = ssim(out.clamp(0,1), gt)

            psnr_values_test.append(psnr_test.item())
            ssim_values_test.append(s_sim_test.item())
            distorted_psnr_test.append(distorted_psnr.item())
            distorted_ssim_test.append(distorted_ssim.item())
            
            #Save image
            if visual==1:
                
                if not os.path.exists(save_images_path):
                    os.makedirs(save_images_path, exist_ok=True)
                    
                io.imsave(os.path.join(save_images_path, 'output_' + str(image_name[0][:-4]) + '_' + \
                          str(model_name) + '_' + str(int(scale)) + '.png'), np.uint8(out[0][0].detach().cpu().numpy().clip(0,1) * 255.))

    print('Test on Poisson noise with peak %d: PSNR %.2f, SSIM %.4f, distorted PSNR %.2f, distorted SSIM %.4f' % (peak, np.array(psnr_values_test).mean(), \
                                                                                              np.array(ssim_values_test).mean(), \
                                                                                              np.array(distorted_psnr_test).mean(), \
                                                                                              np.array(distorted_ssim_test).mean()))

    return

################################################################################################################################

if __name__ == '__main__':
    opt = parser.parse_args()

    print('========= Selected model and inference parameters =============')
    print(opt)
    print('===========================================================================')
    print('\n')

    data_path = './dataset/'
    psf_path  = './dataset/'

    if opt.method=='WFK':
        test_wiener(data_path, psf_path, opt.method, opt.test_scale, opt.model_path, opt.visual, opt.use_gpu)
    if opt.method=='WF_KPN':
        test_wiener(data_path, psf_path, opt.method, opt.test_scale, opt.model_path, opt.visual, opt.use_gpu)
    if opt.method=='WF_KPN_SA':
        test_wienerKPN_SA(data_path, psf_path, opt.method, opt.test_scale, opt.model_path, opt.visual, opt.use_gpu)
    if opt.method == 'WF_UNet':
        test_wienerUnet(data_path, psf_path, opt.method, opt.test_scale, opt.model_path, opt.visual, opt.use_gpu)
    if opt.method == 'UNet':
        test_unet(data_path, psf_path, opt.method, opt.test_scale, opt.model_path, opt.use_gpu, opt.visual)
