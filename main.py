from __future__ import absolute_import, division, print_function
import numpy as np
from argparse import ArgumentParser
import edward as ed
import tensorflow as tf
import matplotlib.pyplot as plt

from learning import learn_latent_model
from create_data import create_data


parser = ArgumentParser()
parser.add_argument('-random_seed', type=int, default=3)
parser.add_argument('-model_type', choices=['standard', 'separated', 'supervised'], default='separated')
parser.add_argument('-estimation_type', choices=['approx_posterior', 'proxy_matching', 'latent_matching'], default='latent_matching')
# In the 'separated' model, the latent variables are  z_x, z_y, z_t, instead of one vector z

parser.add_argument('-n_train', type=int, default=800)  # number of train samples
parser.add_argument('-n_test', type=int, default=500)  # number of test samples

parser.add_argument('-n_epoch', type=int, default=200)

parser.add_argument('-n_neighbours', type=int, default=3) # number of closest neighbours to average in matching

parser.add_argument('-data_file', type=str, default='data.p',
                    help='name of data file to load')
parser.add_argument('-create_new_data', default=False, action="store_true")

args = parser.parse_args()

print(args)
args.show_plots = True
args.save_dataset = True

# ------- Main -------------#
plt.rcParams.update({'font.size': 12})

ed.set_seed(args.random_seed)
np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

# Create Data
train_set, test_set = create_data(args)

# Learning
pehe = learn_latent_model(args, train_set, test_set)

