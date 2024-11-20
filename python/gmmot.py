# DEFINE OT FUNCTIONS
import torch
from ot import emd
import sklearn.mixture
from scipy.optimize import linprog

def sqrtm(M, threshold=1e-6):
    # Returns square root of a SPD matrix with eigenvalue thresholding
    L, V = torch.linalg.eigh(M)
    
    # Apply thresholding to eigenvalues
    L = torch.where(L < threshold, torch.tensor(threshold, dtype=L.dtype), L)  # set small eigenvalues to threshold
    
    # Compute the square roots of the eigenvalues
    L = torch.sqrt(L)
    
    # Reconstruct the matrix square root
    Q = torch.einsum("...jk,...k->...jk", V, L)
    return torch.einsum("...jk,...kl->...jl", Q, torch.transpose(V, -1, -2))

def GaussianMap(mu0, mu1, S0, S1, x):
    # Returns coefficients of Gaussian OT mapping
    cs12 = sqrtm(S0)
    cs12inv = torch.linalg.inv(cs12)

    C = sqrtm(torch.matmul(cs12, torch.matmul(S1, cs12)))
    A = torch.matmul(cs12inv, torch.matmul(C, cs12inv))
    B = mu1 - torch.matmul(mu0, A)
    B = B.unsqueeze(1).expand_as(x)

    return B + torch.matmul(x.T, A).T

def GaussianDist(mu0, mu1, S0, S1):
    # Return OT distance between two gaussians
    S012 = sqrtm(S0)
    S = sqrtm(torch.matmul(S012, torch.matmul(S1, S012)))
    dist = torch.norm(mu0-mu1)**2 + torch.trace(S0 + S1 - 2*S)
    return dist

def GMMMap(pi0, pi1, mu0, mu1, S0, S1):
    # Returns weights of the GW2 map between GMMs and distance
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    M = torch.zeros((K0, K1))
    for k in range(K0):
        for l in range(K1):
            M[k,l]  = GaussianDist(mu0[k,:],mu1[l,:],S0[k,:,:],S1[l,:,:])
    
    wstar     = emd(pi0,pi1,M)
    distGW2   = torch.sum(wstar*M)
    return wstar, distGW2

def GMM(cf, sf, K0, K1, n_init):
    # Returns weights, means and covariances of the GMM fitted to cf
    gmmc = sklearn.mixture.GaussianMixture(n_components=K0, covariance_type='full',n_init=n_init).fit(cf)
    gmms = sklearn.mixture.GaussianMixture(n_components=K1, covariance_type='full',n_init=n_init).fit(sf)

    return torch.from_numpy(gmmc.weights_.real), torch.from_numpy(gmmc.means_.real), torch.from_numpy(gmmc.covariances_.real), torch.from_numpy(gmms.weights_.real), torch.from_numpy(gmms.means_.real), torch.from_numpy(gmms.covariances_.real), torch.from_numpy(gmmc.predict(cf).real), torch.from_numpy(gmmc.predict_proba(cf).real)

def GaussianBarycenterW2(mu,Sigma,alpha,N):
    # Compute the W2 barycenter between several Gaussians
    # mu has size Kxd, with K the number of Gaussians and d the space dimension
    # Sigma has size Kxdxd
    K        = mu.shape[0]  # number of Gaussians
    d        = mu.shape[1]  # size of the space
    Sigman   = torch.eye(d,d)
    mun      = torch.zeros((1,d))
    cost = 0
    
    for n in range(N):
        Sigmandemi = sqrtm(Sigman)
        T = torch.zeros((d,d))
        for j in range(K):
            T+= alpha[j]*sqrtm(Sigmandemi@Sigma[j,:,:]@Sigmandemi)
        Sigman  = T
    
    for j in range(K):
        mun+= alpha[j]*mu[j,:]
    
    for j in range(K):
        cost+= alpha[j]*GaussianDist(mu[j,:],mun,Sigma[j,:,:],Sigman)

    return mun,Sigman,cost

def create_cost_matrix_from_gmm(gmm,alpha,N=10):
    """
    create the cost matrix for the multimarginal problem between all GMM
    create the barycenters (mun,Sn) betweenn all Gaussian components 
    """
    
    nMarginal       = len(alpha)               # number of marginals
    d               = gmm[0][2].shape[1]       # space dimension
    # Create shapes and initialize tensors
    tup = tuple(gmm[k][0] for k in range(nMarginal))
    C = torch.zeros(tup)
    mun = torch.zeros(tup + (d,))
    Sn = torch.zeros(tup + (d, d))

    # Create all possible multi-indices for `tup` dimensions
    multi_indices = torch.cartesian_prod(*[torch.arange(size) for size in tup])

    # Iterate over the multi-indices
    for indices in multi_indices:
        indices = tuple(indices.tolist())  # Convert tensor indices to a tuple
        mu = torch.zeros((nMarginal, d))
        Sigma = torch.zeros((nMarginal, d, d))

        # Compute mu and Sigma for the current indices
        for k in range(nMarginal):
            mu[k, :] = gmm[k][2][indices[k]]
            Sigma[k, :, :] = gmm[k][3][indices[k]]

        # Compute the Gaussian barycenter and update tensors
        mun[indices], Sn[indices], cost = GaussianBarycenterW2(mu, Sigma, alpha, N)
        C[indices] = cost
    
    return C,mun,Sn

def solveMMOT(pi, costMatrix, epsilon = 1e-10):
    """ Author : Alexandre Saint-Dizier
    
    Solver of the MultiMargnal OT problem, using linprog

    Input :
     - pi : list(array) -> weights of the different distributions
     - C : array(d1,...dp) -> cost matrix
     - epsilon : smallest value considered to be positive

    Output :
     - gamma : list of combinaison with positive weight
     - gammaWeights : corresponding weights
    """

    nMarginal = len(pi)
    nPoints = costMatrix.shape

    nConstraints = 0
    nParameters = 1
    for ni in nPoints:
        nConstraints += ni
        nParameters *= ni

    index = 0;
    A = torch.zeros((nConstraints, nParameters))
    b = torch.zeros(nConstraints)
    for i in range(nMarginal):
        ni = nPoints[i]
        b[index:index+ni] = pi[i]
        for k in range(ni):
            Ap = torch.zeros(costMatrix.shape)
            tup = ();
            for j in range(nMarginal):
                if j==i:
                    tup+= (k,)
                else:
                    tup+=(slice(0,nPoints[j]),)
            Ap[tup] = 1
            A[index+k,:]=Ap.flatten()
        index += ni
    A = A.tolist()
    b = b.tolist()
    C = costMatrix.flatten().tolist()

    res = linprog(C, A_eq=A, b_eq =b) #Solve inf <C,X> with constraints AX=b
    gammaWeights = res.x
    gammaWeights = gammaWeights.reshape(costMatrix.shape)
   
    return gammaWeights

def BarycentricInterpolation(gmm, nb_images):

    K0, pi0, mu0, S0 = gmm[0]
    K1, pi1, mu1, S1 = gmm[1]
    K2, pi2, mu2, S2 = gmm[2]
    K3, pi3, mu3, S3 = gmm[3]

    pi   = [pi0,pi1,pi2,pi3]
    d = mu0.shape[1]

    # four corners that will be interpolated by bilinear interpolation
    v1 = torch.tensor((1, 0, 0, 0))
    v2 = torch.tensor((0, 1, 0, 0))
    v3 = torch.tensor((0, 0, 1, 0))
    v4 = torch.tensor((0, 0, 0, 1))

    # result
    gmminterp = []
    Kn        = K0*K1*K2*K3

    for i in range(nb_images):
        for j in range(nb_images):
            tx = float(i) / (nb_images - 1)
            ty = float(j) / (nb_images - 1)

            # weights are constructed by bilinear interpolation
            tmp1 = (1 - tx) * v1 + tx * v2
            tmp2 = (1 - tx) * v3 + tx * v4
            weights = (1 - ty) * tmp1 + ty * tmp2

            if i == 0 and j == 0:
                gmminterp.append([K0,pi0,mu0,S0])
    
            elif i == 0 and j == nb_images - 1:
                gmminterp.append([K2,pi2,mu2,S2])
            
            elif i == nb_images - 1 and j == 0:
                gmminterp.append([K1,pi1,mu1,S1])
            
            elif i == nb_images - 1 and j == (nb_images - 1):
                gmminterp.append([K3,pi3,mu3,S3])
                
            else:            
                C,mun,Sn   = create_cost_matrix_from_gmm(gmm, weights)
                pin        = solveMMOT(pi, C, epsilon = 1e-10).reshape(1,Kn)
                mun,Sn     = mun.reshape(Kn,d),Sn.reshape(Kn,d,d)
                gmminterp.append([Kn,pin,mun,Sn])

    return gmminterp