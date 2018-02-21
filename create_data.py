
from __future__ import absolute_import, division, print_function
import numpy as np




# create synthetic data
def create_data(args):

    # ------- Define Model -------------#
    # Define C model
    n_C = 5  # number of possible values for confounder C
    C_vals = np.arange(0, n_C)  # possible values for confounder C, must be [0,1,....,n_C-1]
    C_vals_prob = np.ones(n_C) / n_C  # probabilities possible values for confounder C

    print('C values & probabilities: ', list(zip(C_vals, C_vals_prob)))

    # Define T model
    # note: we give higher probability for T=1 for larger C's
    treat_prob_per_c = np.linspace(0.1, 0.9, n_C)

    print('Treat prob per C: ', list(zip(C_vals, np.around(treat_prob_per_c, 3))))

    # ------- Generate samples -------------#
    # Generate C
    n_samples = 500
    C = np.random.choice(C_vals, replace=True, size=n_samples, p=C_vals_prob)

    # Generate T
    # T is Bernoulli with p=f(C)
    T = np.less(np.random.rand(n_samples), treat_prob_per_c[C])
    # print('(C,T): ', list(zip(C, T)))

    # Generate E
    E = np.random.rand(n_samples)
    # Generate X
    X = np.random.rand(n_samples, 2)
    X[:, 0] = C + E
    # note: second dimension of X is just noise

    # Generate Y
    Y = C + T

    T = np.expand_dims(T, axis=1)
    Y = np.expand_dims(Y, axis=1)

    X = X.astype(np.float32)
    T = T.astype(np.float32)
    Y = Y.astype(np.float32)

    # print('(X, T, Y): ', list(zip(np.around(X, 3), T, Y)))

    return (X,T,Y)

