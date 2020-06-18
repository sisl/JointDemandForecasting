from CalibratedTimeseriesModels.models.paramunc import *
import torch

from CalibratedTimeseriesModels.dataset import TimeseriesDataset

from CalibratedTimeseriesModels.evaluators import ExplicitEvaluator

from torch.utils import data

import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import expm


from CEEM.ceem.systems import LorenzAttractor, SpringMassDamper


from torch.nn.utils import parameters_to_vector as ptv
from torch.nn.utils import vector_to_parameters as vtp

def main():

    torch.set_default_dtype(torch.float64)

    torch.manual_seed(2)

    # dom = 'Spring'

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

        sig = 0.0005


    prmsig = 0.1
    model = PUEKFGRWModel(sys,
            sig**2 * torch.eye(n), prmsig**2 *torch.eye(3))

    y_inp = torch.randn(1,1,n)

    T = 10
    T1 = T

    dist = model(y_inp, None, T)

    B = 7
    y_out_pred = dist.sample((B,))[...,3:].reshape(B,T,n).detach()

    T = 500
    
    true_prm = ptv(sys.parameters())

    A = torch.eye(3) * prmsig

    nsig = 7
    ytrue = torch.zeros(nsig,T,n)

    for b in range(nsig):
        if b == 6:
            new_params = true_prm
        else:
            if b%2 == 0:
                new_params = true_prm + A[:,b//2] * np.sqrt(3/(1.-0.5))
            else:
                new_params = true_prm - A[:,b//2] * np.sqrt(3/(1.-0.5))
        print(new_params)
        # new_params = true_prm + prmsig * torch.randn_like(true_prm)
        vtp(new_params, sys.parameters())

        with torch.no_grad():
            y0 = y_inp

            ys = [y0]
            for t in range(1,T):
                tinp = torch.tensor([[t]], dtype=torch.get_default_dtype()).repeat(1,1)
                ny = sys.step(tinp, ys[-1]) + sig * torch.randn(1,1,n)
                ys.append(ny)

            ytrue[b:b+1] = torch.cat(ys, dim=1)

    plt.figure(figsize = (12,6))

    for j in range(n):
        plt.subplot(2,n,1+j)
        for i in range(B):
            plt.plot(y_out_pred[i,:,j], alpha=0.5)
            plt.ylabel(r'$x_{}$'.format(j+1))
            plt.xlabel('Time')
            plt.title('Sampled from EKF-GRW')
        plt.plot(dist.loc[:,3:].reshape(T1,n)[:,j].detach(), linewidth=2.0, color='r')

    for j in range(n):
        plt.subplot(2,n,n+1+j)
        for i in range(nsig):
            plt.plot(ytrue[i,:,j], alpha=0.5)
            plt.xlabel('Time')
            plt.ylabel(r'$x_{}$'.format(j+1))
            plt.title('True Simulation')
        # plt.plot(dist.loc.reshape(T,n)[:,j], linewidth=2.0, color='r')

    plt.tight_layout()

    plt.savefig('./sandbox/kmenda/EKF-GRW_%s.png'%dom,dpi=300)

    plt.show()
    

if __name__ == '__main__':
    main()