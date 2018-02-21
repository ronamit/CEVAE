from __future__ import absolute_import, division, print_function
import numpy as np
from argparse import ArgumentParser
import edward as ed
import tensorflow as tf

from learning import learn_latent_model
from create_data import create_data


parser = ArgumentParser()
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-model-type', choices=['standard', 'separated'], default='separated')
# In the 'separated' model, the latent variables are  z_x, z_y, z_t, instead of one vector z
parser.add_argument('-n_epoch', type=int, default=500)
args = parser.parse_args()


# ------- Main -------------#

ed.set_seed(args.random_seed)
np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

(X,T,Y) = create_data(args)

learn_latent_model(args, X, T, Y)

