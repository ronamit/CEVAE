from __future__ import absolute_import, division, print_function
import numpy as np
from argparse import ArgumentParser
import edward as ed
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle, os
from copy import deepcopy
import timeit, time

from learning import learn_latent_model
from create_data import create_data


parser = ArgumentParser()
parser.add_argument('-random_seed', type=int, default=3)
# parser.add_argument('-model_type', choices=['standard', 'separated', 'supervised'], default='separated')
# parser.add_argument('-estimation_type', choices=['approx_posterior', 'proxy_matching', 'latent_matching'], default='latent_matching')
# In the 'separated' model, the latent variables are  z_x, z_y, z_t, instead of one vector z

# parser.add_argument('-n_train', type=int, default=800)  # number of train samples
parser.add_argument('-n_test', type=int, default=1000)  # number of test samples

parser.add_argument('-n_epoch', type=int, default=200)

parser.add_argument('-n_neighbours', type=int, default=3) # number of closest neighbours to average in matching

parser.add_argument('-data_file', type=str, default='data.p',
                    help='name of data file to load')
# parser.add_argument('-create_new_data', default=True, action="store_true")

general_args = parser.parse_args()

print(general_args)

args = deepcopy(general_args)

args.create_new_data = True
args.show_plots = False
args.save_dataset = False

# ------- Main -------------#
plt.rcParams.update({'font.size': 12})

ed.set_seed(args.random_seed)
np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)


save_file = 'Results.p'
load_results = False

if not load_results:

    # (method_name, model_type, estimation_type)
    methods_opt = [('standard CEVAE - posterior', 'standard', 'approx_posterior'),
                   ('separated CEVAE - posterior', 'separated', 'approx_posterior'),
                   ('standard CEVAE - matching', 'standard', 'latent_matching'),
                   ('separated CEVAE - matching', 'separated', 'latent_matching'),
                   ('Proxy matching', '', 'proxy_matching')]

    n_rep = 10
    # n_rep = 1 # DEBUG*******************

    num_train_samp_grid = np.arange(100, 1100, step=100)
    # num_train_samp_grid = np.arange(100,300, step=100)  # DEBUG*******************

    n_samp_grid = len(num_train_samp_grid)
    n_methods = len(methods_opt)
    err_mat = np.zeros((n_methods, n_samp_grid, n_rep))

    start_time = timeit.default_timer()

    for i_method, method_t in enumerate(methods_opt):
        args.model_type = method_t[1]
        args.estimation_type = method_t[2]

        for i_samp_num, samp_num in enumerate(num_train_samp_grid):
            args.n_train = samp_num

            for i_rep in range(n_rep):
                print('Method {} out of {}'.format(i_method + 1, n_methods))
                print('Sample size {} out of {}'.format(i_samp_num + 1, n_samp_grid))
                print('Replication {} out of {}'.format(i_rep+1, n_rep))

                # Create Data
                train_set, test_set = create_data(args)

                # Learning
                err_mat[i_method, i_samp_num, i_rep] = learn_latent_model(args, train_set, test_set)

    pickle.dump({'general_args': general_args, 'methods_opt': methods_opt,
                 'err_mat': err_mat, 'num_train_samp_grid': num_train_samp_grid}, open(save_file, "wb"))

    stop_time = timeit.default_timer()
    print('Total runtime: ' +
                 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)))
else:
    saved_data = pickle.load(open(save_file, "rb"))
    methods_opt = saved_data['methods_opt']
    num_train_samp_grid = saved_data['num_train_samp_grid']
    err_mat = saved_data['err_mat']



# Show results
plt.figure()

for i_method, method_t in enumerate(methods_opt):
    n_rep = err_mat.shape[2]
    err_curr = err_mat[i_method]
    avg_err = err_curr.mean(axis=1)
    sem_err = err_curr.std(axis=1) / np.sqrt(n_rep)

    plt.errorbar(num_train_samp_grid, avg_err, yerr=sem_err, label=method_t[0])
    plt.xlabel('Number of training samples')
    plt.ylabel('PEHE')

plt.legend()
plt.show()
