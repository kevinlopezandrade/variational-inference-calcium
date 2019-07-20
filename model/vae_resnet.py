import torch 
import torch.nn as nn
import torch.nn.functional as F


class RecognitionModel(nn.Module):

    def __init__(self, n_latent):
        super().__init__()

        conv1 = nn.Sequential(
            nn.Conv2d(1, 20, 3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )

        residual_layers = nn.Sequential(
            ResidualBlock(20),
            ResidualProjection(20, 30),
            ResidualBlock(30),
            ResidualProjection(30, 40),
            ResidualBlock(40),
            ResidualProjection(40, 50),
            ResidualProjection(50, 50)
        )

        fc = nn.Sequential(
            nn.Linear(5000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU()
        )

        mean = nn.Sequential(
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, n_latent)
        )

        log_variance = nn.Sequential(
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, n_latent)
        )

        mean_concat = nn.Sequential(
            nn.Linear(n_latent + 2, n_latent + 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_latent + 2),
            nn.Linear(n_latent + 2, n_latent)
        )

        log_variance_concat = nn.Sequential(
            nn.Linear(n_latent + 2, n_latent + 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_latent + 2),
            nn.Linear(n_latent + 2, n_latent)
        )

        self.conv1 = conv1
        self.residual_layers = residual_layers
        self.fc = fc
        self.mean = mean
        self.log_variance = log_variance
        self.latent_dimension = n_latent
        self.mean_concat = mean_concat
        self.log_variance_concat = log_variance_concat
            

    def forward(self, N, X):

        N = self.conv1(N)
        N = self.residual_layers(N)

        N = N.view(-1, 5000)
        N = self.fc(N)

        mean = self.mean(N)
        log_variance = self.log_variance(N)

        joint_mean = torch.cat((mean, X), dim=1)
        joint_log_variance = torch.cat((log_variance, X), dim=1)

        mean_concat = self.mean_concat(joint_mean)
        log_variance_concat = self.log_variance_concat(joint_log_variance)

        return mean_concat, log_variance_concat



class ObservationModel(nn.Module):

    def __init__(self, n_latent):
        super().__init__()

        fc = nn.Sequential(
            nn.Linear(n_latent, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 5000),
        )

        residual_layers = nn.Sequential(
            ResidualProjectionTrasnpose(50, 50, 0),
            ResidualProjectionTrasnpose(50, 40, 0),
            ResidualBlockTranspose(40),
            ResidualProjectionTrasnpose(40, 30, 1),
            ResidualBlockTranspose(30),
            ResidualProjectionTrasnpose(30, 20, 1),
            ResidualBlockTranspose(20),
        )

        conv_layers = nn.Sequential(
            nn.ConvTranspose2d(20, 1, 3, stride=1, padding=0),
        )

        self.fc = fc
        self.residual_layers = residual_layers
        self.conv_layers = conv_layers 


    def forward(self, Z):
        X = self.fc(Z)

        X = X.view(X.shape[0], 50, 10, 10)

        X = self.residual_layers(X)
        X = self.conv_layers(X)

        return X



class PriorModel(nn.Module):

    def __init__(self, n_latent):
        super().__init__()

        mean = nn.Sequential(
            nn.Linear(2, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent)
        )

        log_variance = nn.Sequential(
            nn.Linear(2, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent)
        )

        self.mean = mean
        self.log_variance = log_variance

    def forward(self, X):

        mean = self.mean(X)
        log_variance = self.log_variance(X)

        return mean, log_variance


class VAE(nn.Module):

    def __init__(self, n_latent):
        super().__init__()

        self.recognition_model = RecognitionModel(n_latent)
        self.observation_model = ObservationModel(n_latent)
        self.prior_model       = PriorModel(n_latent)

    def forward(self, X):
        pass



#=====================================================
#============ Auxialiary clases (layers) =============
#=====================================================

class ResidualProjection(nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_in,  n_out, 3, stride=2, padding=1),
            nn.BatchNorm2d(n_out),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_out)
        )

        self.projection = nn.Conv2d(n_in, n_out, 3, stride=2, padding=1)

    def forward(self, X):

        identity = X

        X = self.conv_layers(X)

        Y = X + self.projection(identity)

        Y = F.relu(Y)

        return Y


class ResidualBlock(nn.Module):

    def __init__(self, n_in):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_in, n_in, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_in),
            nn.ReLU(),
            nn.Conv2d(n_in, n_in, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_in)
        )

    def forward(self, X):

        identity = X

        X = self.conv_layers(X)

        Y = X + identity

        Y = F.relu(Y)

        return Y

class ResidualProjectionTrasnpose(nn.Module):

    def __init__(self, n_in, n_out, output_padding):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(n_in,  n_out, 3, stride=2, padding=1, output_padding=output_padding),
            nn.BatchNorm2d(n_out),
            nn.ReLU(),
            nn.ConvTranspose2d(n_out, n_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_out)
        )

        self.projection = nn.ConvTranspose2d(n_in, n_out, 3, stride=2, padding=1, output_padding=output_padding)

    def forward(self, X):

        identity = X

        X = self.conv_layers(X)

        Y = X + self.projection(identity)

        Y = F.relu(Y)

        return Y


class ResidualBlockTranspose(nn.Module):

    def __init__(self, n_in):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(n_in, n_in, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_in),
            nn.ReLU(),
            nn.ConvTranspose2d(n_in, n_in, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_in)
        )

    def forward(self, X):

        identity = X

        X = self.conv_layers(X)

        Y = X + identity

        Y = F.relu(Y)

        return Y
