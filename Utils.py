

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

def get_fc_layer_fn(l2_reg_scale=1e-4):
    reg = tf.contrib.layers.l2_regularizer(l2_reg_scale) # weight decay regularization
    return lambda input, out_dim, activation: dense_layer(
        input, out_dim, activation, regularizer=reg)
#----------------------------------------------------------------------------------------------------------------------------#

def evalaute_effect_estimate(est_y0, est_y1, test_set, method_name=''):

  est_cate = est_y1 - est_y0
  ate_est = np.mean(est_cate)
  print('Estimated ATE:', ate_est)

  true_cate = test_set['Y1'] - test_set['Y0']
  rmse = np.sqrt(np.mean(np.square(true_cate - est_cate)))
  print('CATE estimation RMSE: ', rmse)

  # plot scatter H vs. CATE
  H = test_set['H']
  plt.scatter(H.flatten(), est_cate.flatten(), label='Estimated - ' + method_name)
  plt.scatter(H.flatten(), true_cate.flatten(), label='Ground Truth')
  plt.xlabel('H')
  plt.ylabel('CATE')
  plt.legend()
  plt.show(framealpha=0.5)

  # ----------------------------------------------------------------------------------------------------------------------------#