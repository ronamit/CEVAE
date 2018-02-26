from __future__ import absolute_import, division, print_function
import numpy as np
from argparse import ArgumentParser
import edward as ed
import tensorflow as tf

from learning import learn_latent_model
from create_data import create_data


parser = ArgumentParser()
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-model_type', choices=['standard', 'separated', 'supervised'], default='supervised')
parser.add_argument('-estimation_type', choices=['approx_posterior', 'proxy_matching', 'latent_matching'], default='latent_matching')
# In the 'separated' model, the latent variables are  z_x, z_y, z_t, instead of one vector z

parser.add_argument('-n_train', type=int, default=1000)  # number of train samples
parser.add_argument('-n_test', type=int, default=600)  # number of test samples

parser.add_argument('-n_epoch', type=int, default=200)


args = parser.parse_args()

print(args)

# ------- Main -------------#

ed.set_seed(args.random_seed)
np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

# Create Data
train_set, test_set = create_data(args)

# Learning
learn_latent_model(args, train_set, test_set)

