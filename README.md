# IWAE
chainer implementation of [IWAE](https://arxiv.org/abs/1509.00519 "IWAE").
This code is inspired [code1](https://github.com/Ma-sa-ue/practice/tree/master/generative_model) and [code2](https://github.com/chainer/chainer/tree/master/examples/vae).

## Abstract
By utilizing *k* sampling, IWAE reduces the ELBO.
If *k* is *1*, it corresponds to the standard VAE.

## How to use
python iwae.py --sampling_number=k　　

## Analysis

I check the value of ELBO after training.  

|*k*=1|*k*=5|*k*=10|
|:--:|:--:|:--:|
|100.14|98.88|98.34|

By increasing the value of *k*, you can reduce ELBO.  

The detail is described in [iwae_analysis.ipynb](https://github.com/smayru/IWAE/blob/master/IWAE_analysis.ipynb).
