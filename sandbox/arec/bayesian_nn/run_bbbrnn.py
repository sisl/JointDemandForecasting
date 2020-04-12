import torch
from torchsummary import summary
import argparse
import numpy as np
from codebase.models.BBBTimeSeriesPredModel import BBBTimeSeriesPredModel
from codebase.train import train
import codebase.utils as ut
import data.data_utils as data_ut

# Data
batch_size = 20
n_batches = 8000 # used by dummy data
n_input_steps = 30
n_pred_steps = 10
input_feat_dim = 2
pred_feat_dim = 2
dataset_name = 'dummy{}d'.format(input_feat_dim)

# Network
hidden_feat_dim = 50

# Model
cell = 'LSTM'
BBB = True
sharpen = False
likelihood_cost_form = 'gaussian'
# likelihood_cost_form = 'mse'
nlayers = 1
dropout = 0
pi = 0.25
std1 = np.exp(-1)
std2 = np.exp(-6)

# Train
dev_mode = False
training = True
lr = 2
run = 0
iter_max = 400000
iter_plot = 1000
# # automatic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = False if device == torch.device('cpu') else True

# Enforced settings:
if not BBB:
    sharpen = False
    layout = [
        ('model={:s}', cell),
        ('BBB={:s}', str(BBB)),
        ('data={:s}', dataset_name),
        ('nlayers={:d}', nlayers),
        ('nhid={:d}', hidden_feat_dim),
        ('dropout={:.1f}', dropout),
        ('loss={:s}', likelihood_cost_form),
        ('run={:d}', run),
    ]
else:
    layout = [
        ('model={:s}', cell),
        ('BBB={:s}', str(BBB)),
        ('data={:s}', dataset_name),
        ('nlayers={:d}', nlayers),
        ('nhid={:d}', hidden_feat_dim),
        ('dropout={:.1f}', dropout),
        ('loss={:s}', likelihood_cost_form),
        ('sharpen={:s}', str(sharpen)),
        ('pi={:.2f}', pi),
        ('logstd1={:d}', int(np.log(std1))),
        ('logstd2={:d}', int(np.log(std2))),
        ('run={:d}', run),
    ]

model_name = '_'.join([t.format(v) for (t, v) in layout])
print(model_name)

if not dev_mode:
    ut.prepare_dirs(model_name, overwrite_existing=True)
else:
    iter_plot = np.inf

training_set = data_ut.dummy_data_creator(
        batch_size=batch_size, 
        n_batches=n_batches, 
        input_feat_dim=input_feat_dim,
        n_input_steps=n_input_steps, 
        n_pred_steps=n_pred_steps,
        kernel=data_ut.sinusoidal_kernel,
        device=device)

model = BBBTimeSeriesPredModel(
        num_rnn_layers=nlayers,
        pi=pi,
        std1=std1,
        std2=std2,
        gpu=gpu,
        BBB=BBB,
        training=training,
        sharpen=sharpen,
        dropout=dropout,
        likelihood_cost_form=likelihood_cost_form,
        input_feat_dim=input_feat_dim,
        pred_feat_dim=pred_feat_dim,
        hidden_feat_dim=hidden_feat_dim,
        n_input_steps=n_input_steps,
        n_pred_steps=n_pred_steps,
        rnn_cell_type=cell,
        name=model_name,
        device=device).to(device)

train(model, training_set, batch_size, n_batches, device, 
        kernel=data_ut.sinusoidal_kernel,
        lr=1e-3,
        clip_grad=5,
        iter_max=iter_max, 
        iter_save=np.inf, 
        iter_plot=iter_plot, 
        reinitialize=False)

