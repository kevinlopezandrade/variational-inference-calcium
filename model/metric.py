import torch


""" KLD(p_1 | p_2)"""
def KLD2(mean_1, mean_2, log_std_1, log_std_2):

    variance_1 = torch.exp(log_std_1) ** 2
    variance_2 = torch.exp(log_std_2) ** 2

    kld = (log_std_2 - log_std_1) + (1/(2*variance_2) * (variance_1 + (mean_1 - mean_2)**2)) - 1/2

    kld = torch.sum(kld)

    return kld

def gaussian_observation_loss(X, mean_X):

    X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
    mean_X = mean_X.view(-1, mean_X.shape[1] * mean_X.shape[2] * mean_X.shape[3])

    difference = X - mean_X

    squared = difference ** 2 
    loss_xi = torch.sum(squared, dim=1)

    loss_X = -0.5 * torch.sum(loss_xi)

    return loss_X


def KLD(mean, log_sigma):
    """ I return the kl loss"""

    mean_squared = mean ** 2
    sigma_squared = torch.exp(log_sigma) ** 2
    log_sigma_squared = 2 * log_sigma

    divergence_X = -0.5 * torch.sum(1 + log_sigma_squared - mean_squared - sigma_squared)


    return divergence_X


def bernouli_observation_loss(X, y):

    # Numerical Stability if we add 1e-12 to avoid log of zero
    log_likelihood_xi = torch.sum(torch.log((y ** X)*((1 - y)**(1-X))), dim=1)

    log_likelihood_X  = torch.sum(log_likelihood_xi)

    if torch.isnan(log_likelihood_X):
        breakpoint()

    return log_likelihood_X

def bernouli_observation_loss_ae(X, y):

    log_likelihood_xi = torch.sum(X * torch.log(y) + (1 - X) * torch.log(1 - y), dim=1)

    log_likelihood_X  = torch.sum(log_likelihood_xi)

    M = X.shape[0]

    return log_likelihood_X*(1/M)
