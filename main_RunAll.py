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
parser.add_argument('-n_test', type=int, default=10000)  # number of test samples

parser.add_argument('-n_epoch', type=int, default=50)

parser.add_argument('-n_neighbours', type=int, default=3) # number of closest neighbours to average in matching

parser.add_argument('-data_file', type=str, default='data.p',
                    help='name of data file to load')
# parser.add_argument('-create_new_data', default=True, action="store_true")

parser.add_argument('-data_type', default='UriToy', type=str)  #  'UriToy' /  'RonToy'

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
sess = tf.InteractiveSession()
tf.set_random_seed(args.random_seed)


save_file = 'Results.p'
run_mode = 'LoadComplete' # 'RunExp' / 'LoadComplete' / 'LoadTempFiles'


if run_mode == 'RunExp':

    # (method_name, model_type, estimation_type)
    methods_opt = [('standard CEVAE - posterior', 'standard', 'approx_posterior'),
                   ('separated CEVAE - posterior', 'separated', 'approx_posterior'),
                   ('separated+confounder CEVAE - posterior', 'separated_with_confounder', 'approx_posterior'),
                   ('standard CEVAE - matching', 'standard', 'latent_matching'),
                   ('separated CEVAE - matching', 'separated', 'latent_matching'),
                   ('Proxy matching', '', 'proxy_matching')]

    methods_opt = [('standard CEVAE - posterior', 'standard', 'approx_posterior'),
                   ('separated CEVAE - posterior', 'separated', 'approx_posterior'),
                   ('separated+confounder CEVAE - posterior', 'separated_with_confounder', 'approx_posterior')]

    n_methods = len(methods_opt)

    methods_to_run = range(n_methods)
    # methods_to_run = [1] # DEBUG*******************

    n_rep = 20
    # n_rep = 1 # DEBUG*******************

    num_train_samp_grid = np.arange(100, 1100, step=100)
    # num_train_samp_grid = np.arange(100,300, step=100)  # DEBUG*******************

    n_samp_grid = len(num_train_samp_grid)

    err_mat = -1 * np.ones((n_methods, n_samp_grid, n_rep))

    start_time = timeit.default_timer()

    for i_method in methods_to_run:
        method_t = methods_opt[i_method]
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
        # save temp results file
        pickle.dump({'general_args': general_args, 'methods_opt': methods_opt,
                     'err_mat': err_mat, 'num_train_samp_grid': num_train_samp_grid},
                    open(save_file+'_Temp_{}'.format(i_method), "wb"))

    pickle.dump({'general_args': general_args, 'methods_opt': methods_opt,
                 'err_mat': err_mat, 'num_train_samp_grid': num_train_samp_grid}, open(save_file, "wb"))

    stop_time = timeit.default_timer()
    print('Total runtime: ' +
                 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)))

elif run_mode == 'LoadComplete':
    saved_data = pickle.load(open(save_file, "rb"))
    methods_opt = saved_data['methods_opt']
    num_train_samp_grid = saved_data['num_train_samp_grid']
    err_mat = saved_data['err_mat']


elif run_mode == 'LoadTempFiles':
    i_method = 0
    temp_file = save_file + '_Temp_{}'.format(i_method)
    saved_data = pickle.load(open(save_file, "rb"))
    methods_opt = saved_data['methods_opt']
    num_train_samp_grid = saved_data['num_train_samp_grid']
    err_mat = saved_data['err_mat']
    n_methods = len(methods_opt)
    for i_method in range(n_methods):
        temp_file = save_file + '_Temp_{}'.format(i_method)
        saved_data = pickle.load(open(temp_file, "rb"))
        err_mat_temp = saved_data['err_mat']
        err_mat[i_method] = err_mat_temp[i_method]


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
