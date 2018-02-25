
from __future__ import absolute_import, division, print_function
import numpy as np




# create synthetic data
def create_data(args):

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
    E = -0.5 + 1 * np.random.rand(n_samples)
    # Generate X
    X = np.random.rand(n_samples, 3)
    X[:, 0] = H + E
    X[:, 1] = np.random.rand(n_samples) * 10
    X[:, 2] = np.random.rand(n_samples) * 10
    # note: second dimension of X is just noise

    # Generate Y
    Y1 = 0.1 * W - 1 + 4 * np.less(5 * np.ones(n_samples), H)  # for T=1
    Y0 = 0.1 * W  # for T=0
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

    return train_set, test_set















# create synthetic data
# def create_data(args):
#
#     # ------- Define Model -------------#
#     # Define C model
#     n_C = 5  # number of possible values for confounder C
#     C_vals = np.arange(0, n_C)  # possible values for confounder C, must be [0,1,....,n_C-1]
#     C_vals_prob = np.ones(n_C) / n_C  # probabilities possible values for confounder C
#
#     print('C values & probabilities: ', list(zip(C_vals, C_vals_prob)))
#
#     # Define T model
#     # note: we give higher probability for T=1 for larger C's
#     treat_prob_per_c = np.linspace(0.1, 0.9, n_C)
#
#     print('Treat prob per C: ', list(zip(C_vals, np.around(treat_prob_per_c, 3))))
#
#     # ------- Generate samples -------------#
#     # Generate C
#     n_samples = 500
#     C = np.random.choice(C_vals, replace=True, size=n_samples, p=C_vals_prob)
#
#     # Generate T
#     # T is Bernoulli with p=f(C)
#     T = np.less(np.random.rand(n_samples), treat_prob_per_c[C])
#     # print('(C,T): ', list(zip(C, T)))
#
#     # Generate E
#     E = np.random.rand(n_samples)
#     # Generate X
#     X = np.random.rand(n_samples, 2)
#     X[:, 0] = C + E
#     # note: second dimension of X is just noise
#
#     # Generate Y
#     Y = C + T
#
#     T = np.expand_dims(T, axis=1)
#     Y = np.expand_dims(Y, axis=1)
#
#     X = X.astype(np.float32)
#     T = T.astype(np.float32)
#     Y = Y.astype(np.float32)
#
#     # print('(X, T, Y): ', list(zip(np.around(X, 3), T, Y)))
#
#     return (X,T,Y)
#
