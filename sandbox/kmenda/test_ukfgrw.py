from CalibratedTimeseriesModels.models.ukfgrw import *
import torch

from CalibratedTimeseriesModels.dataset import TimeseriesDataset

from CalibratedTimeseriesModels.evaluators import ExplicitEvaluator

from torch.utils import data

import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import expm


from CEEM.ceem.systems import LorenzAttractor, SpringMassDamper

def main():

    torch.set_default_dtype(torch.float64)

    torch.manual_seed(2)

    dom = 'Spring'

    dom = 'Lorenz'

    if dom == 'Spring':

        n = 4

        M = D = K = torch.tensor([[1., 2.], [2., 5.]])

        D = torch.eye(2) * 2.0

        dt = 0.1

        method = 'midpoint'

        sys = SpringMassDamper(M, D, K, dt, method=method)

        sig = 0.01

    else:

        n = 3

        sigma = torch.tensor([10.])
        rho = torch.tensor([28.])
        beta = torch.tensor([8. / 3.])

        C = torch.randn(2, 3)

        dt = 0.04

        sys = LorenzAttractor(sigma, rho, beta, C, dt, method='rk4')

        sig = 0.05


    model = UKFGRWModel(sys,
            sig**2 * torch.eye(n), w0=0.)

    y_inp = torch.randn(1,1,n)

    T = 250

    with torch.no_grad():
        dist = model(y_inp, None, T)

    B = 10
    y_out_pred = dist.sample((B,)).reshape(B,T,n).detach()

    with torch.no_grad():
        y0 = y_inp.repeat(B,1,1)

        ys = [y0]
        for t in range(1,T):
            tinp = torch.tensor([[t]], dtype=torch.get_default_dtype()).repeat(B,1)
            ny = sys.step(tinp, ys[-1]) + sig * torch.randn(B,1,n)
            ys.append(ny)

        ytrue = torch.cat(ys, dim=1)

    plt.figure(figsize = (12,6))

    for j in range(n):
        plt.subplot(2,n,1+j)
        for i in range(B):
            plt.plot(y_out_pred[i,:,j], alpha=0.5)
            plt.ylabel(r'$x_{}$'.format(j+1))
            plt.xlabel('Time')
            plt.title('Sampled from UKF-GRW')
        plt.plot(dist.loc.reshape(T,n)[:,j], linewidth=2.0, color='r')

    for j in range(n):
        plt.subplot(2,n,n+1+j)
        for i in range(B):
            plt.plot(ytrue[i,:,j], alpha=0.5)
            plt.xlabel('Time')
            plt.ylabel(r'$x_{}$'.format(j+1))
            plt.title('True Simulation')
        # plt.plot(dist.loc.reshape(T,n)[:,j], linewidth=2.0, color='r')

    plt.tight_layout()

    plt.savefig('./sandbox/kmenda/UKF-GRW_%s.png'%dom,dpi=300)

    plt.show()
    

if __name__ == '__main__':
    main()