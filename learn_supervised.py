from __future__ import absolute_import, division, print_function
import numpy as np

import edward as ed
import matplotlib.pyplot as plt
import tensorflow as tf
from edward.models import Normal, Bernoulli

from Utils import batch_generator, evalaute_effect_estimate, get_fc_layer_fn, matching_estimate
# ----------------------------------------------------------------------------------------------------------------------------#
def learn_supervised(args, train_set, test_set):

    # Parameters
    n_hidd = 512  # number of hidden units
    n_epoch = args.n_epoch
    learning_rate = 0.001
    batch_size = 128

    fc_layer = get_fc_layer_fn(l2_reg_scale=1e-4)

    x_train, t_train, y_train = train_set['X'], train_set['T'], train_set['Y']

    n_train = x_train.shape[0]
    x_dim = x_train.shape[1]

    batch_size = min(batch_size, n_train)

    # ------ Define Graph ---------------------#

    # ------ Define Inputs
    # define placeholder which will receive data batches
    x_ph = tf.placeholder(tf.float32, [None, x_dim])
    t_ph = tf.placeholder(tf.float32, [None, 1])
    y_ph = tf.placeholder(tf.float32, [None, 1])

    n_ph = tf.shape(x_ph)[0]  # number of samples fed to placeholders

    # ------  regression with a neural-network model y=NN(x,t)
    input = tf.concat([x_ph, t_ph], axis=1)
    hidden_layer = fc_layer(input, n_hidd, tf.nn.elu)
    hidden_layer = fc_layer(hidden_layer, n_hidd, tf.nn.elu)
    net_out = fc_layer(hidden_layer, 1, None)
    cost = tf.reduce_mean((net_out - y_ph)**2)

    # ------ Training

    batch_size = min(batch_size, n_train)
    n_iter_per_epoch = n_train // batch_size

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # end graph def

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(n_epoch):
            train_generator = batch_generator(np.random.permutation(n_train), batch_size)
            avg_loss = 0.0
            for j in range(n_iter_per_epoch):
                # Take batch:
                idx = next(train_generator)
                x_b, t_b, y_b = x_train[idx], t_train[idx], y_train[idx]
                feed_dict = {x_ph: x_b, t_ph: t_b, y_ph: y_b}
                _, curr_cost = sess.run([optimizer, cost], feed_dict=feed_dict)

                avg_loss +=  curr_cost
            avg_loss = avg_loss / n_iter_per_epoch
            avg_loss = avg_loss / batch_size
            if epoch % 50 == 0:
                print('Epoch {}, avg loss {}'.format(epoch, avg_loss))

        # ------ Evaluation -

        x_test = test_set['X']

        # estimate CATE epr sample:
        forced_t = np.ones((args.n_test, 1))
        est_y1 = sess.run([net_out], feed_dict={x_ph: x_test, t_ph: forced_t})[0]
        est_y0 = sess.run([net_out], feed_dict={x_ph: x_test, t_ph: 0*forced_t})[0]

        evalaute_effect_estimate(est_y0, est_y1, test_set, model_name='supervised', estimation_type='')
    # end session


