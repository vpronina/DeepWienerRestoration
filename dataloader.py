from PIL import Image
import numpy as np
import torch
import json
from torch.utils.data import Dataset
import os
from skimage import io
from scipy.signal import convolve2d as conv2
import random
from scipy import interpolate

def pil_loader(path):
    img = Image.open(path)
    return img

class CellDataset(Dataset):

    def __init__(self, root, psf_path, noise_mode, scale,
                 std, types=None, loader=pil_loader):
        """
            root (string): Directory with all the images
            psf_path (string): Directory with all PSFs
            noise_mode (string): Mode of noise to model (Gaussian or Poisson)
            scale (float): Maximum intensity peak for rescaling to model
            different Poisson noise levels
            std (float): Standard deviation of Gaussian noise for the case of
            Gaussian noise
            types (list/string): If not None, then select names of cell types to work with
        """
        super().__init__()
        all_types = ['Confocal_BPAE_B', 'Confocal_BPAE_G', 'Confocal_BPAE_R',
                     'Confocal_MICE', 'FYVE_HeLa', 'TwoPhoton_BPAE_B',
                     'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_R', 'TwoPhoton_MICE',
                     'WideField_BPAE_G', 'WideField_BPAE_R']
        if types is None:
            self.types = all_types
        else:
            assert all([img_type in all_types for img_type in types])
            self.types = types
        
        self.root = root
        self.psf_path = psf_path
        self.noise_mode = noise_mode
        self.loader = loader
        self.scale = scale
        self.std = std
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'test',
                        '# samples': len(self.samples)}

        print(json.dumps(dataset_info, indent=4))

    def __len__(self):
        return len(self.samples)

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        subdir = os.path.join(root_dir, 'Test')

        lst0 = os.listdir(subdir)
        lst0.sort()
        for types_subdir in lst0:
            lst = os.listdir(os.path.join(subdir, types_subdir))
            lst.sort()
            for name in lst:
                if types_subdir in self.types:
                        img_path = os.path.join(subdir, types_subdir, name)
                        samples.append([img_path, name])

        return samples

    def __getitem__(self, index):
        np.random.seed(0) #for reproducibility
        gt_path = self.samples[index][0]
        image_name = self.samples[index][1]
        gt = self.loader(gt_path)
        gt = np.array(gt).astype(float)
        idx = (index % 5) + 1
        if self.noise_mode=='gaussian':
            psf = io.imread(os.path.join(self.psf_path, 'PSF_gaussian', 'PSF' + str(idx) + '.tif'))
        if self.noise_mode=='poisson':
            psf = io.imread(os.path.join(self.psf_path, 'PSF_poisson', 'PSF' + str(idx) + '.tif'))
            
        psf = np.array(psf).astype(float)
        psf = psf / np.sum(psf)

        peak = self.scale #For Gaussian noise case scale = 1

        gt1 = gt - gt.min()
        gt2 = gt1 / gt1.max()
        gt = gt2 * peak

        std = self.std #For Poisson noise case std = 0

        image = conv2(gt, psf[::-1, ::-1], mode='same', boundary='symm')

        if self.noise_mode == 'gaussian':
            noise = np.random.normal(0, std, size=(image.shape[-1], image.shape[-2]))
            image += noise

        if self.noise_mode == 'poisson':
            image = np.random.poisson(image)

        sample = [gt, image]

        gt = sample[0].copy()
        image = sample[1].copy()

        sample[0] = torch.from_numpy(gt)
        sample[1] = torch.from_numpy(image)
        sample[0] = sample[0].float()
        sample[1] = sample[1].float()
        return sample, psf, idx, image_name, peak, std

    def __len__(self):
        return len(self.samples)