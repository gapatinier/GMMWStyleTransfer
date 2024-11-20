# DEFINE TRANSFORMATIONS IN THE FEATURE SPACE
import torch
from python import gmmot
import numpy as np

def wct(cf, sf):
    # WHITENING AND COLORING TRANSFORM
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)

    c_mean = torch.mean(cfv, 1)
    c_mean = c_mean.unsqueeze(1).expand_as(cfv)
    cfv = cfv - c_mean
    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1)

    w = torch.linalg.inv(gmmot.sqrtm(c_covm))
    whitened = torch.mm(w, cfv)

    sf = sf.double()
    _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(c_channels, -1)

    s_mean0 = torch.mean(sfv, 1)
    s_mean = s_mean0.unsqueeze(1).expand_as(sfv)
    sfv = sfv - s_mean

    s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)
    c = gmmot.sqrtm(s_covm)


    colored = torch.mm(c, whitened) + s_mean0.unsqueeze(1).expand_as(cfv)
    target_features = colored.view_as(cf)

    return target_features

def gaussian_st(cf, sf):
    # GAUSSIAN STYLE TRANSFORM
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)

    c_mean = torch.mean(cfv, 1)
    c_mean0 = c_mean.unsqueeze(1).expand_as(cfv)
    cfv = cfv - c_mean0

    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1)

    sf = sf.double()
    _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(c_channels, -1)

    s_mean = torch.mean(sfv, 1)
    s_mean0 = s_mean.unsqueeze(1).expand_as(sfv)
    sfv = sfv - s_mean0

    s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)

    target_features = gmmot.GaussianMap(c_mean, s_mean, c_covm, s_covm, cfv)
    target_features = target_features.view_as(cf)

    return target_features


def gmm_st(cf, sf, K0=5, K1=5, transport=None):
    # GMM STYLE TRANSFORM 
    cf = cf.double()
    cfv = cf.view(cf.size(0), -1)

    sf = sf.double()
    sfv = sf.view(sf.size(0), -1)

    pi0, mu0, S0, pi1, mu1, S1, ClassesC, ProbaClassesC = gmmot.GMM(cfv.T.detach(), sfv.T.detach(), K0, K1, n_init=3)

    wstar, dist = gmmot.GMMMap(pi0/torch.sum(pi0), pi1/torch.sum(pi1), mu0, mu1, S0, S1)

    T = torch.zeros((K0, K1, cfv.size(0), cfv.size(1)))
    for k in range(K0):
        for l in range(K1):
            T[k,l,:,:] = gmmot.GaussianMap(mu0[k,:], mu1[l,:], S0[k,:,:], S1[l,:,:], cfv)
    
    Tpush = torch.zeros((cfv.size(0), cfv.size(1)))

    if transport == 'mean':
        for k in range(K0):
            for l in range(K1):
                Tpush += wstar[k,l]/pi0[k]*ProbaClassesC[:,k].T*T[k,l,:,:]
    elif transport == 'rand':
        tmp = torch.zeros((K0*K1,cfv.size(1)))
        for k in range(K0):
            for l in range(K1):
                tmp[k+K0*l,:]= wstar[k,l]/pi0[k]*ProbaClassesC[:,k]
        for i in range(cfv.size(1)):
            proba_sum = torch.sum(tmp[:, i])
            if proba_sum > 0:  # Check if the sum is greater than zero to avoid division by zero
                prob = tmp[:, i] / proba_sum  # Normalize the probabilities
            else:
                # Handle the case when sum is zero (e.g., assign equal probabilities)
                prob = torch.ones_like(tmp[:, i]) / (K0 * K1)
            # Convert to numpy array before passing to np.random.choice
            prob = prob.numpy()
            # Make sure the probabilities sum to 1
            prob /= prob.sum()
            # Sample from the probabilities
            m = np.random.choice(K0 * K1, p=prob)
            l = m // K0
            k = m - K0 * l
            Tpush[:,i] = T[k,l,:,i]
    else:
        normalisation = torch.zeros((cfv.size(1)))
        for k in range(K0):
            for l in range(K1):
                Tpush += wstar[k,l]*T[k,l,:,:]*(ClassesC==k).T
                normalisation +=wstar[k,l]*(ClassesC==k).T
        Tpush = Tpush/normalisation
    
    Tpush = Tpush.view_as(cf)

    return Tpush, ClassesC, ProbaClassesC