

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#----------------------------------------------------------------------------------------------------------------------------#
def batch_generator(array, batch_size):
  """Generate batch with respect to array's first axis."""
  start = 0  # pointer to where we are in iteration
  while True:
    stop = start + batch_size
    diff = stop - array.shape[0]
    if diff <= 0:
      batch = array[start:stop]
      start += batch_size
    else:
      batch = np.concatenate((array[start:], array[:diff]))
      start = diff
    yield batch

#----------------------------------------------------------------------------------------------------------------------------#
def dense_layer(input, out_dim, activation, regularizer):
    return tf.layers.dense(input, out_dim, activation=activation,
                           kernel_regularizer=regularizer, bias_regularizer=regularizer)

def get_fc_layer_fn(l2_reg_scale=1e-4, depth=1):
    reg = tf.contrib.layers.l2_regularizer(l2_reg_scale) # weight decay regularization
    return lambda input, out_dim, activation: run_layers(input, out_dim, activation, reg, depth)

def run_layers(input, out_dim, activation, reg, depth):
    x = input
    for i_layer in range(depth):
        x = dense_layer(x, out_dim, activation, regularizer=reg)
    return x

#-------------------------------------------------------------------------
# ---------------------------------------------------#

def evalaute_effect_estimate(est_y0, est_y1, test_set, args, model_name, estimation_type):
    est_cate = est_y1 - est_y0
    est_cate = np.squeeze(est_cate)
    print('Estimated ATE:', np.mean(est_cate))

    true_cate = test_set['Y1'] - test_set['Y0']

    # Precision in Estimation of Heterogeneous Effect:
    pehe = np.mean(np.square(true_cate - est_cate))
    print('CATE mean squared estimation error (PEHE): ', pehe)
    # pehe1 = np.mean(np.abs(true_cate - est_cate))
    # print('CATE mean abs estimation error (PEHE-L1): ', pehe1)

    if args.show_plots:
        # plot scatter H vs. CATE
        H = test_set['H']
        plt.scatter(H.flatten(), est_cate.flatten(), label='Estimated', marker='.')
        plt.scatter(H.flatten(), true_cate.flatten(), label='Ground Truth', marker='o', facecolors='None', edgecolors='g', s=80, alpha=0.05)
        plt.xlabel('H')
        plt.ylabel('CATE')
        plt.legend()
        plt.title('Model: {} \n Estimation: {}'.format(model_name, estimation_type))
        plt.show()

    return pehe



# ----------------------------------------------------------------------------------------------------------------------------#
def matching_estimate(feat_train, t_train, y_train, feat_test, n_neighbours=5):
    n_test = feat_test.shape[0]
    n_train =  feat_train.shape[0]
    est_y0 = np.zeros((n_test, 1))
    est_y1 = np.zeros((n_test, 1))

    # penalty vectors:
    train_idx = np.arange(n_train)
    t_train = t_train.flatten()
    train_idx0 = train_idx[t_train == 0]
    feat_train0 = feat_train[t_train == 0]
    train_idx1 = train_idx[t_train == 1]
    feat_train1 = feat_train[t_train == 1]

    # For each test sample
    for i_sample, feat_sample in enumerate(feat_test):
        # Find closest train samples with t=0
        dists = np.linalg.norm(feat_sample - feat_train0, axis=1)
        closests_idx = np.argsort(dists)[:n_neighbours]
        closests_idx = train_idx0[closests_idx]
        est_y0[i_sample] = y_train[closests_idx].mean()

        # Find closest train samples with t=1
        dists = np.linalg.norm(feat_sample - feat_train1, axis=1)
        closests_idx = np.argsort(dists)[:n_neighbours]
        closests_idx = train_idx1[closests_idx]
        est_y1[i_sample] = y_train[closests_idx].mean()



    return est_y0, est_y1