
from __future__ import absolute_import, division, print_function
import numpy as np
import os
import matplotlib as plt
import pickle


# create synthetic data
def create_data(args):

    np.random.seed(args.random_seed)

    if  args.create_new_data or args.data_file == '' or not os.path.exists(args.data_file):
        print('Creating new data file: ', args.data_file)
    elif args.save_dataset:
        saved_data = pickle.load(open(args.data_file, "rb"))
        print('Loading data file: ', args.data_file)
        return saved_data['train_set'], saved_data['test_set']

    n_train = args.n_train
    n_test = args.n_test
    n_samples = n_train + n_test

    # ------- Define Model -------------#
    # Define W model
    n_W = 9  # number of possible values for confounder W
    W_vals = np.arange(0, n_W)  # possible values for confounder W, must be [0,1,....,nW-1]
    W_vals_prob = np.ones(n_W) / n_W  # probabilities possible values for confounder C

    # Define T model
    # note: we give higher probability for T=1 for larger W's
    treat_prob_per_W = np.linspace(0.1, 0.9, n_W)

    # ------- Generate samples -------------#
    # Generate W
    W = np.random.choice(W_vals, replace=True, size=n_samples, p=W_vals_prob)

    # Generate H (uniform [0,10])
    H = np.random.rand(n_samples) * 10

    # Generate T
    # T is Bernoulli with p=f(W)
    T = np.less(np.random.rand(n_samples), treat_prob_per_W[W])

    # Generate E uniform [-1,1]
    # Generate X

    # A = np.eye(2,2) # + np.random.rand(2,2)
    # v = np.matmul(A, np.array([H, W]))
    # X = -0.5 + 1 * np.random.rand(n_samples, 4)
    # X[:, 0] += v[0]
    # X[:, 1] += v[1]
    # X[:, 2] *= 3
    # X[:, 3] *= 3
    # # X[:, 2] = np.random.rand(n_samples)


    # Generate E uniform [-1,1]
    E1 = -0.5 + 1 * np.random.rand(n_samples)
    E2 = -0.5 + 1 * np.random.rand(n_samples)
    E3 = -0.5 + 1 * np.random.rand(n_samples)
    E4 = -0.5 + 1 * np.random.rand(n_samples)
    # Generate X
    X = np.random.rand(n_samples, 3)
    X[:, 0] = H * 0 + W * 1 + E1
    X[:, 1] = H * 1 + W * 0 + E2
    X[:, 2] = E3 * 10
    # X[:, 3] = E4 * 10
    # note: second dimension of X is just noise

    # Generate Y
    #
    # # Experiment A
    # Y1 = 0 * W - 1 + 2 * np.less(5 * np.ones(n_samples), H)  # for T=1
    # Y0 = 0 * W  # for T=0

    # Experiment B
    # Y1 = 0. * W + H * np.ones(n_samples)  # for T=1
    # Y0 = 0. * W  # for T=0

    # Experiment C
    Y1 = 0. * W + H * np.ones(n_samples) + np.less(5 * np.ones(n_samples), H) * (H - 5) # for T=1
    Y0 = 0. * W + H * np.ones(n_samples)  # for T=0

    Y = T * Y1 + (1 - T) * Y0  # measured Y

    # change dimension to [n_sample X 1]
    T = np.expand_dims(T, axis=1)
    Y = np.expand_dims(Y, axis=1)

    # print('(X, T, Y): ', list(zip(np.around(X, 3), T, Y)))

    X = X.astype(np.float32)
    T = T.astype(np.float32)
    Y = Y.astype(np.float32)

    train_set = dict()
    train_set['X'] = X[:n_train]
    train_set['T'] = T[:n_train]
    train_set['Y'] = Y[:n_train]

    test_set = dict()
    test_set['X'] = X[n_train:]
    test_set['T'] = T[n_train:]
    test_set['Y'] = Y[n_train:]
    test_set['H'] = H[n_train:]
    test_set['W'] = W[n_train:]
    test_set['Y0'] = Y0[n_train:]
    test_set['Y1'] = Y1[n_train:]

    pickle.dump({'train_set': train_set, 'test_set': test_set}, open(args.data_file, "wb"))


    # # CATE
    # H = test_set['H']
    # true_cate = test_set['Y1'] - test_set['Y0']
    # plt.scatter(H.flatten(), true_cate.flatten(), label='Ground Truth')
    # plt.xlabel('H')
    # plt.ylabel('CATE')
    # plt.legend()
    # plt.show()


    return train_set, test_set


