import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
import pandas as pd

import os
opj = os.path.join

def main():

    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)

    x0 = torch.randn(1000,2)

    cov = 0.1 * torch.tensor([[1., 0.25],[0.25, 0.5]])
    mu = torch.zeros(2)

    dist = mvn(mu, cov)

    T = 100

    xs = torch.zeros(1000,T,2)

    xs[:,0] = x0

    for t in range(1,T):
        xs[:,t] = xs[:,t-1] + dist.sample((1000,))

    xs = xs.detach()
    t = torch.arange(T).unsqueeze(0).repeat(1000,1)

    torch.save({'t': t, 'y': xs}, 'datasets/gauss_rw.data')

    import matplotlib.pyplot as plt

    plt.subplot(2,1,1)
    plt.plot(xs[:,:,0].T)
    plt.subplot(2,1,2)
    plt.plot(xs[:,:,1].T)

    plt.show()


if __name__ == '__main__':
    main() 