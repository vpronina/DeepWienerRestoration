{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Models:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We provide 10 pretrained models: 5 for poisson noise case and 5 for gaussian noise case. \n",
    "In detail, models are:\n",
    "- UNet\n",
    "- Wiener filter with learnable identical kernels\n",
    "- Wiener filter with predictable kernels per-image\n",
    "- Wiener filter with predictable kernels per-pixel\n",
    "- Deconvolution with predictable gradient of regularizer per-image"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To evaluate pre-trained Wiener filter with learnable identical kernels, run the following command:"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "python main.py --mode test --method WFK --test_model --test_scale --test_std --model_path ./models/ --visual 1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Training:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To train Wiener filter with learnable identical kernels, run the following command:"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "python main.py --mode train --method WFK --nfilt 8 --filtsize 3 --filters_init dct --noise_mode poisson --epochs 300 --save_model 1 --name MY_MODEL --continue_training 0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To train Wiener filter with predictable kernels per-image, run the following command:"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "python main.py --mode train --method WF_KPN --nfilt 8 --filtsize 3 --noise_mode poisson --epochs 300 --save_model 1 --name MY_MODEL --continue_training 0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To train Wiener filter with predictable kernels per-pixel, run the following command:"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "python main.py --mode train --method WF_KPN_SA --noise_mode poisson --epochs 300 --save_model 1 --name MY_MODEL --continue_training 0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To train Deconvolution model with predictable gradient of regularizer per-image (WF_UNet), run the following command"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "python main.py --mode train --method WF_UNet --noise_mode poisson --iter 10 --epochs 300 --save_model 1 --name MY_MODEL --continue_training 0"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
