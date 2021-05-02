import torch
from collections import OrderedDict
import os
from modules import*
from networks import UNet
from utils import pad_psf_shape, EdgeTaper


def model_load(noise_mode, method, model_path):

    if method == 'WFK':
        model_name = 'WFK_' + str(noise_mode)
        model = WienerFilter(n_filters=8, filter_size=3)
        
    if method == 'WF_KPN':
        model_name = 'WF_KPN_' + str(noise_mode)
        model = WienerFilterKPN(n_filters=8, filter_size=3)
        
    if method == 'WF_UNet':
        model_name = 'WF_UNet_' + str(noise_mode)
        model = WienerFilter_UNet() 
        
    if method == 'WF_KPN_SA':
        model_name = 'WF_KPN_SA_' + str(noise_mode)
        
        if noise_mode=='gaussian':
            model = Wiener_KPN_SA(maxiter=250, tol=1e-6) 
        if noise_mode == 'poisson':
            model = Wiener_KPN_SA(maxiter=200, tol=1e-4) 

    state_dict = torch.load(os.path.join(model_path, model_name))

    state_dict = state_dict['model_state_dict']  
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        name = key[7:]  
        new_state_dict[name] = val
    model.load_state_dict(new_state_dict)
       
    return model

def edgetaper(image, psf):
    
    image = pad_psf_shape(image, psf)
    for l in range(image.shape[0]):
        image[l] = EdgeTaper.apply(image[l], psf[l][0])
                                                    
    return image