import torch
import wandb
import time

from os import path
from time import strftime

from model import metric
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from datetime import datetime, timedelta


def train(recognition_model, observation_model, optimizer, batches, alfa, device, logger):

    observation_model.train()
    recognition_model.train()

    latent_dimension = recognition_model.latent_dimension

    normal = MultivariateNormal(torch.zeros(latent_dimension), torch.eye(latent_dimension))

    total_elbo           = 0.0
    total_kld            = 0.0
    total_log_likelihood = 0.0
    total_mse            = 0.0

    print("Annealing with {}".format(alfa))

    for X in batches:

        optimizer.zero_grad()

        X = X.to(device)
        M = X.shape[0]

        # Generate Z
        mean, log_variance = recognition_model(X)
        std = torch.sqrt(torch.exp(log_variance))
        log_std = torch.log(std)

        e = normal.sample(sample_shape=(M,)).to(device)
        Z = mean + std * e

        # Compute Observation Model
        mean_X = observation_model(Z)

        # Compute Kullback Divergence
        KLD = alfa * (1/M * metric.KLD(mean, log_std))
        log_likelihood = 1/M * metric.gaussian_observation_loss(X, mean_X)
        mse = F.mse_loss(mean_X, X)

        elbo = -1*KLD + log_likelihood

        # Backropagate
        (-elbo).backward()

        optimizer.step()

        total_elbo += (elbo.item() * M)
        total_kld  += (KLD.item() * M)
        total_log_likelihood += (log_likelihood.item() * M)
        total_mse += (mse.item() * M)


    return total_elbo, total_kld, total_log_likelihood, total_mse

def test(recognition_model, observation_model, batches, device, callback, logger):

    recognition_model.eval()
    observation_model.eval()

    latent_dimension = recognition_model.latent_dimension

    normal = MultivariateNormal(torch.zeros(latent_dimension), torch.eye(latent_dimension))

    total_elbo           = 0.0
    total_kld            = 0.0
    total_log_likelihood = 0.0
    total_mse            = 0.0


    with torch.no_grad():

        for X in batches:

            X = X.to(device)
            M = X.shape[0]

            # Generate Z
            mean, log_sigma_squared = recognition_model(X)
            sigma = torch.sqrt(torch.exp(log_sigma_squared))
            log_sigma = torch.log(sigma)

            e = normal.sample(sample_shape=(M,)).to(device)
            Z = mean + sigma * e

            # Compute Observation Model
            mean_X = observation_model(Z)

            # Compute Kullback Divergence
            KLD = 1*(1/M * metric.KLD(mean, log_sigma))
            log_likelihood = 1/M * metric.gaussian_observation_loss(X, mean_X)

            # Since we minimize
            elbo = -1*KLD + log_likelihood

            total_elbo += (elbo.item() * M)
            total_kld  += (KLD.item() * M)
            total_log_likelihood += (log_likelihood.item() * M)
            total_mse += F.mse_loss(mean_X, X).item() * M


        if callback is not None:
            callback(recognition_model, observation_model, X, mean_X)


    return total_elbo, total_kld, total_log_likelihood, total_mse


class Inference:
    """Abstract clas for variational inference"""

    def __init__(self, model, optimizer, device, logger):

        self.model             = model
        self.recognition_model = model.recognition_model
        self.observation_model = model.observation_model
        self.optimizer         = optimizer
        self.device            = device
        self.logger            = logger

        # None intializations
        self.hyperparameters = None
        self.risks = None


    def optimize(self, epochs, data_loader, validate=False, test_loader=None, callback=None):

        recognition_model = self.recognition_model
        observation_model = self.observation_model

        optimizer = self.optimizer
        device    = self.device
        logger    = self.logger

        # Nasty hack to more clean code
        batch_size = data_loader.batch_size
        N = len(data_loader.dataset)

        if validate:
            N_test = len(test_loader.dataset)

        try:
            print("Start training")
            for epoch in range(1, epochs+1):
                start_time = time.perf_counter()


                alfa = 1

                metrics = train(recognition_model, observation_model, optimizer, data_loader, alfa, device, logger)

                end_time = time.perf_counter()

                average_elbo           = metrics[0]/N
                average_kld            = metrics[1]/N
                average_log_likelihood = metrics[2]/N
                average_mse            = metrics[3]/N


                print("\nEpoch {:5} computed in {:4.3f} s".format(epoch, end_time - start_time))

                print("ELBO: {}".format(average_elbo))
                print("KLD: {}".format(average_kld))
                print("LogLike: {}".format(average_log_likelihood))

                wandb.log({'ELBO Train': average_elbo}, commit=False)
                wandb.log({'KLD Train': average_kld}, commit=False)
                wandb.log({'Log Likelihood Train' : average_log_likelihood}, commit=False)
                wandb.log({'MSE Train': average_mse}, commit=False)

                if validate:
                    test_metrics = test(recognition_model, observation_model, test_loader, device, callback, logger)

                    average_elbo_test           = test_metrics[0]/N_test
                    average_kld_test            = test_metrics[1]/N_test
                    average_log_likelihood_test = test_metrics[2]/N_test
                    average_mse_test            = test_metrics[3]/N_test

                    wandb.log({'ELBO Test': average_elbo_test}, commit=False)
                    wandb.log({'KLD Test': average_kld_test}, commit=False)
                    wandb.log({'Log Likelihood Test' : average_log_likelihood_test}, commit=False)
                    wandb.log({'MSE Test': average_mse_test}, commit=False)

                else:
                    average_elbo_test = None


                wandb.log()


            # Once the training has finsihed save the hyperparameters
            hyperparameters = {
                    'epochs': epochs,
                    'last_epoch': epoch,
                    'batch_size': batch_size,
                    'optimizer': optimizer.state_dict()
            }

            self.hyperparameters = hyperparameters

            risks = {
                    'elbo': average_elbo,
                    'test_elbo': average_elbo_test
            }

            self.risks = risks

        # Handle excpetion and save the las computed values
        except KeyboardInterrupt:
            print("Training paused, saving last epoch")
            hyperparameters = {
                    'epochs': epochs,
                    'last_epoch': epoch - 1,
                    'batch_size': batch_size,
                    'optimizer': optimizer.state_dict()
            }

            self.hyperparameters = hyperparameters

            risks = {
                    'elbo': average_elbo,
                    'test_elbo': average_elbo_test
            }

            self.risks = risks


    def save(self, directory):

        if self.hyperparameters is None:
            raise ValueError("Nothing to save, the model was not trained")
        
        model = self.model
        hyperparameters = self.hyperparameters
        risks = self.risks

        if model.__class__.__name__ == "DataParallel":
            model = model.module

        recognition_model = model.recognition_model
        observation_model = model.observation_model

        time_format = '%d-%m-%Y-%H:%M:%S'
        date = datetime.utcnow() + timedelta(hours=2)

        name = "{}-{}.pt".format(model.__class__.__name__, date.strftime(time_format))
        file_path = path.join(directory, name)

        to_save = {
                'recognition_model': recognition_model.state_dict(),
                'observation_model': observation_model.state_dict(),
                'hyperparameters': hyperparameters,
                'risks': risks
        }

        torch.save(to_save, file_path)

        print("Model saved as {}".format(name))
