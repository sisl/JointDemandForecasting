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

    n = 3

    sigma = torch.tensor([10.])
    rho = torch.tensor([28.])
    beta = torch.tensor([8. / 3.])

    C = torch.randn(2, 3)

    dt = 0.04

    sys = LorenzAttractor(sigma, rho, beta, C, dt, method='rk4')


    # n = 4

    # M = D = K = torch.tensor([[1., 2.], [2., 5.]])

    # D = torch.eye(2) * 0.001

    # dt = 0.1

    # method = 'midpoint'

    # sys = SpringMassDamper(M, D, K, dt, method=method)

    model = UKFGRWModel(sys,
            0.0005*torch.eye(n))

    y_inp = torch.randn(1,1,n)

    dist = model(y_inp, None, 100)

    B = 10
    y_out_pred = dist.sample((10,)).reshape(10,100,n).detach()

    # y_out_pred = model(y_inp, None, 1, 20)[0]

    # evaluator = ExplicitEvaluator()

    # metrics = evaluator(model, test_d)

    for i in range(10):
        # plt.plot(y_out_pred[i,:,0],y_out_pred[i,:,1], alpha=0.5)

        plt.plot(y_out_pred[i,:,0], alpha=0.5)

    plt.show()
    

if __name__ == '__main__':
    main()