import sys
import utils
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model.inference_conditional import Inference
from model.datasets import ConditionalDatasetPrunned
from model.vae_resnet import VAE
from utils import PlotCallbackVAE


if __name__ == "__main__":


    parameters = sys.argv

    _, epochs, batch_size, learning_rate = parameters

    epochs        = int(epochs)
    batch_size    = int(batch_size)
    learning_rate = float(learning_rate)

    # Start wandb
    wandb.init()
    wandb.config.epochs = epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        in_cluster  = True
        num_workers = 8
        pin_memory  = False
    else:
        in_cluster  = False
        num_workers = 0
        pin_memory = False

    data_train = ConditionalDatasetPrunned("data/X_train_crop.npy", "data/Y_embedded_train.npy", in_cluster)
    data_test  = ConditionalDatasetPrunned("data/X_test_crop.npy",  "data/Y_embedded_test.npy", in_cluster)

    batch_view_train = DataLoader(data_train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    batch_view_test  = DataLoader(data_test,  shuffle=True, batch_size=batch_size, num_workers=num_workers)

    print("Data loaded")

    net = VAE(60)
    wandb.watch(net)

    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Create the learner
    inference = Inference(net, optimizer, device, None)

    callback = PlotCallbackVAE(device)

    # Fit the model
    inference.optimize(epochs, batch_view_train, validate=True, test_loader=batch_view_test, callback=callback)

    # Save the model if in cluster

    if in_cluster:
        inference.save("saved_runs")
