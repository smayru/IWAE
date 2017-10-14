#!/usr/bin/env python
from __future__ import print_function
import argparse

import matplotlib.pyplot as plt
import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import data

import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=100, type=int,
                    help='number of epochs to learn')
parser.add_argument('--sampling_number', '-sa', default=5, type=int,
                    help='value of sampling number')
parser.add_argument('--dimz', '-z', default=20, type=int,
                    help='dimention of encoded vector')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--test', action='store_true',
                    help='Use tiny datasets for quick tests')
args = parser.parse_args()

class weighted_VAE(chainer.Chain):
    """Weighted importance VAE"""

    def __init__(self, n_in, n_latent,sampling_number, n_h):
        super(weighted_VAE, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_in)
        self.sampling_number=sampling_number

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def get_loss_func(self,x, C=1.0):

        batchsize = len(self.encode(x)[0])
        z=list()
        mu, ln_var = self.encode(x)
        for l in six.moves.range(self.sampling_number):
            z.append(F.gaussian(mu, ln_var))
        for iii in range(self.sampling_number):

            if iii==0:
                
                rec_loss=0
                z = F.gaussian(mu, ln_var)
                rec_loss += F.sum(F.bernoulli_nll(x, self.decode(z, sigmoid=False), reduce='no'),axis=1)/(batchsize)
                loss=rec_loss+F.sum(C * gaussian_kl_divergence(mu, ln_var,reduce='no'),axis=1)/ batchsize
                loss=F.reshape(loss,[batchsize,1])

            else:
                rec_loss=0
                z = F.gaussian(mu, ln_var)
                rec_loss += F.sum(F.bernoulli_nll(x, self.decode(z, sigmoid=False), reduce='no'),axis=1)/(batchsize)
                tmp_loss=rec_loss+F.sum(C * gaussian_kl_divergence(mu, ln_var,reduce='no'),axis=1)/ batchsize
                tmp_loss=F.reshape(tmp_loss,[batchsize,1])
                loss=F.concat((loss,tmp_loss),axis=1)
        importance_weight = F.softmax(loss)
        self.total_loss=F.sum(importance_weight*loss)
        return self.total_loss

batchsize = args.batchsize
n_epoch = args.epoch
n_latent = args.dimz

print('GPU: {}'.format(args.gpu))
print('# dim z: {}'.format(args.dimz))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data(args.test)
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

if args.test:
    N = 30
else:
    N = 60000

x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

sampling_number=args.sampling_number

# Prepare weighted_VAE model
model = weighted_VAE(784, n_latent,sampling_number, 500)
if args.gpu >= 0:
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)



# Learning loop
training_error=np.zeros([n_epoch,2])
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_loss = 0       # total loss
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        model.zerograds()
        loss=model.get_loss_func(x)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(x.data)

    print('train mean loss={}'
          .format(sum_loss / N))
    training_error[epoch-1,:]=[epoch,sum_loss]

error_name="training_error_sampling_number_"+str(args.sampling_number)+".csv"
np.savetxt(error_name,training_error,delimiter=',',header="Epoch,Error")

# Save the model and the optimizer
print('save the model')
model_name="weigted_vae_sampling_number_"+str(args.sampling_number)+".model"
serializers.save_npz(model_name, model)
state_name="weigted_vae_sampling_number_"+str(args.sampling_number)+".state"
print('save the optimizer')
serializers.save_npz(state_name, optimizer)
model.to_cpu()

# original images and reconstructed images
def save_images(x, filename):
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(28, 28))
    fig.savefig(filename)

train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
x = chainer.Variable(np.asarray(x_train[train_ind]))
with chainer.no_backprop_mode():
    x1 = model(x)
save_images(x.data, 'train')
save_images(x1.data, 'train_reconstructed_sampling_num_'+str(args.sampling_number))

test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
x = chainer.Variable(np.asarray(x_test[test_ind]))
with chainer.no_backprop_mode():
    x1 = model(x)
save_images(x.data, 'test')
save_images(x1.data, 'test_reconstructed_sampling_num_'+str(args.sampling_number))


# draw images from randomly sampled z
z = chainer.Variable(np.random.normal(0, 1, (9, n_latent)).astype(np.float32))
x = model.decode(z)
save_images(x.data, 'sampled_sampling_num_'+str(args.sampling_number))

