import math
import time
import torch
import numpy as np
import sys
import wandb
import matplotlib.pyplot as plt
import plotly

from os import path
from time import strftime
from torch.distributions import MultivariateNormal


def transfer_saved_model(model_instance, state_dict, inference=False, strict=True):
    """Loads a saved model:
    inference: If we don't want to compute the gradients wrt to some parameter of the model
    by setting to true this argument we don't record the operations of the loaded model

    By default in pytorch all the parameters of the modules have the requires_grad set to true
    
    """

    model_instance.load_state_dict(state_dict, strict=strict)

    if inference:
        # Change recursively the required grad for all the parameters
        for param in model_instance.parameters(recurse=True):
            param.requires_grad = False


    return model_instance


def load_saved_run(model_path):

    saved_dict = None

    if torch.cuda.is_available():
        saved_dict = torch.load(model_path)
    else:
        saved_dict = torch.load(model_path, map_location='cpu')

    return saved_dict


def output_size_convolution_layer(H, W, kernel_size, stride, padding, dilation=1):
    """
    Returns the ouput size of a squared image after convoling
    with the specified parameters
    """

    numerator = H + ( 2 * padding) - (dilation * (kernel_size - 1)) - 1
    denominator = stride

    return math.floor((numerator/denominator) + 1)


def output_size_deconvolution_layer(H, W, kernel_size, stride, padding, output_padding, dilation=1):

    H_out = (H - 1) * stride - (2 * padding) + (dilation * (kernel_size - 1)) + output_padding + 1

    return H_out

def save(model, directory):

    model = model

    if model.__class__.__name__ == "DataParallel":
        model = model.module


    time_format = '%d-%m-%Y-%H:%M:%S'
    name = "{}-{}.pt".format(model.__class__.__name__, strftime(time_format))
    file_path = path.join(directory, name)

    to_save = {
            'model': model.state_dict(),
    }

    torch.save(to_save, file_path)

    print("Model saved as {}".format(name))

class LogCallback:

    def __init__(self, device):
        self.device = device

    def __call__(self, *args):
        raise NotImplementedError

class PlotCallbackVAE(LogCallback):

    def __call__(self, recognition_model, observation_model, X, Y_):

        device  = self.device

        with torch.no_grad():

            M = X.shape[0]

            random_index = torch.randint(M, size=(1,))[0].item()

            sample = X[random_index].clone().cpu()
            reconstructed_sample = Y_[random_index].clone().cpu()

            img_input = wandb.Image(sample, caption="Input")
            wandb.log({"Input": img_input}, commit=False)

            img_reconstruction = wandb.Image(reconstructed_sample, caption="Reconstruction")
            wandb.log({"Reconstruction": img_reconstruction}, commit=False)

class PlotRegressionEmbedding(LogCallback):

    def __init__(self, device):
        super().__init__(device)
        self.decoder = torch.load("decoder.pt")
        self.decoder.eval()

    def __call__(self, model, X, Y, Y_):

        model.eval()
        decoder = self.decoder

        with torch.no_grad():
            M = X.shape[0]

            random_index = torch.randint(M, size=(1,))[0].item()

            expected  = Y[random_index].clone().cpu()

            approximation = Y_[random_index].clone().cpu()

            fig = plt.figure()
            axe = fig.subplots(1,1)

            axe.scatter(x=-expected[1], y=expected[0], c="black")
            axe.scatter(x=-approximation[1], y=approximation[0], c="red")
            axe.set_xlim(-3.5, 3.5)
            axe.set_ylim(-3.5, 3.5)

            wandb.log({'Regression': fig}, commit=False)

            img_reconstruction = wandb.Image(decoder(approximation.view(1, 2))[0].view(64, 64), caption="Reconstruction")
            wandb.log({"Reconstruction": img_reconstruction}, commit=False)


            img_expected_stimulus = wandb.Image(decoder(expected.view(1, 2))[0].view(64, 64), caption="Expected Reconstruction")
            wandb.log({"Expected Reconstruction" : img_expected_stimulus}, commit=False)
