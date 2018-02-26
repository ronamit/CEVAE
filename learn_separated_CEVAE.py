from __future__ import absolute_import, division, print_function
import numpy as np

import edward as ed
import matplotlib.pyplot as plt
import tensorflow as tf
from edward.models import Normal, Bernoulli

from Utils import batch_generator, evalaute_effect_estimate, get_fc_layer_fn, matching_estimate
# ----------------------------------------------------------------------------------------------------------------------------#


def learn_separated(args, train_set, test_set):

    # Parameters
    n_hidd = 1024  # number of hidden units per layer
    n_epoch = args.n_epoch
    learning_rate = 0.001
    batch_size = 128

    hidden_layer = get_fc_layer_fn(l2_reg_scale=1e-4, depth=1)
    out_layer = get_fc_layer_fn(l2_reg_scale=1e-4)

    x_train, t_train, y_train = train_set['X'], train_set['T'], train_set['Y']

    n_train = x_train.shape[0]
    x_dim = x_train.shape[1]

    batch_size = min(batch_size, n_train)

    # ------ Define Graph ---------------------#

    # ------ Define Inputs ---------------------#
    # define placeholder which will receive data batches
    x_ph = tf.placeholder(tf.float32, [None, x_dim])
    t_ph = tf.placeholder(tf.float32, [None, 1])
    y_ph = tf.placeholder(tf.float32, [None, 1])

    n_ph = tf.shape(x_ph)[0]  # number of samples fed to placeholders

    # ------ Define generative model /decoder-----------------------#

    # z_x_dim = 1
    z_t_dim = 1
    z_y_dim = 1
    # latent_dims = (z_x_dim, z_t_dim, z_y_dim)
    latent_dims = (z_t_dim, z_y_dim)
    # prior over latent variables:
    # p(zx) -
    # zx = Normal(loc=tf.zeros([n_ph, z_x_dim]), scale=tf.ones([n_ph, z_x_dim]))
    # p(zt) -
    zt = Normal(loc=tf.zeros([n_ph, z_t_dim]), scale=tf.ones([n_ph, z_t_dim]))
    # p(zy) -
    zy = Normal(loc=tf.zeros([n_ph, z_y_dim]), scale=tf.ones([n_ph, z_y_dim]))


    # p(x|z) - likelihood of proxy X
    # z = tf.concat([zx, zt, zy], axis=1)
    z = tf.concat([zt, zy], axis=1)
    hidden = hidden_layer(z, n_hidd, tf.nn.elu)
    x = Normal(loc=out_layer(hidden, x_dim, None),
               scale=out_layer(hidden, x_dim, tf.nn.softplus),
               name='gaussian_px_z')

    # p(t|zt)
    hidden = hidden_layer(zt, n_hidd, tf.nn.elu)
    probs = out_layer(hidden, 1, tf.nn.sigmoid)  # output in [0,1]
    t = Bernoulli(probs=probs, dtype=tf.float32, name='bernoulli_pt_z')

    # p(y|t,zy)
    hidden = hidden_layer(zy, n_hidd, tf.nn.elu)  # shared hidden layer
    mu_y_t0 = out_layer(hidden, 1, None)
    mu_y_t1 = out_layer(hidden, 1, None)
    # y = Normal(loc=t * mu_y_t1 + (1. - t) * mu_y_t0, scale=tf.ones_like(mu_y_t0))
    sigma_y_t0 = out_layer(hidden, 1, tf.nn.softplus)
    sigma_y_t1 = out_layer(hidden, 1, tf.nn.softplus)
    y = Normal(loc=t * mu_y_t1 + (1. - t) * mu_y_t0,
               scale=t * sigma_y_t1 + (1. - t) * sigma_y_t0)


    # ------ Define inference model - CEVAE variational approximation (encoder)

    # q(t|x)
    hqt = hidden_layer(x_ph, n_hidd, tf.nn.elu)
    probs = out_layer(hqt, 1, tf.nn.sigmoid)  # output in [0,1]
    qt = Bernoulli(probs=probs, dtype=tf.float32)

    # q(y|x,t)
    hqy = hidden_layer(x_ph, n_hidd, tf.nn.elu)  # shared hidden layer
    mu_qy_t0 = out_layer(hqy, 1, None)
    mu_qy_t1 = out_layer(hqy, 1, tf.nn.elu)
    sigma_qy_t1 = out_layer(hqy, 1, tf.nn.softplus)
    sigma_qy_t0 = out_layer(hqy, 1, tf.nn.softplus)
    # qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))
    qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0,
                scale=qt * sigma_qy_t1 + (1. - qt) * sigma_qy_t0)


    # # q(z_x|x,t,y)
    # inpt2 = tf.concat([x_ph, qy], axis=1)
    # hqz = hidden_layer(inpt2, n_hidd, tf.nn.elu)  # shared hidden layer
    # muq_t0 = out_layer(hqz, z_x_dim, None)
    # sigmaq_t0 = out_layer(hqz, z_x_dim, tf.nn.softplus)
    # muq_t1 = out_layer(hqz, z_x_dim, None)
    # sigmaq_t1 = out_layer(hqz, z_x_dim, tf.nn.softplus)
    # qzx = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
    #             scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

    # q(z_t|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = out_layer(inpt2, n_hidd, tf.nn.elu)  # shared hidden layer
    muq_t0 = out_layer(hqz, z_t_dim, None)
    sigmaq_t0 = out_layer(hqz, z_t_dim, tf.nn.softplus)
    muq_t1 = out_layer(hqz, z_t_dim, None)
    sigmaq_t1 = out_layer(hqz, z_t_dim, tf.nn.softplus)
    qzt = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
                scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

    # q(z_y|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = hidden_layer(inpt2, n_hidd, tf.nn.elu)  # shared hidden layer
    muq_t0 = out_layer(hqz, z_y_dim, None)
    sigmaq_t0 = out_layer(hqz, z_y_dim, tf.nn.softplus)
    muq_t1 = out_layer(hqz, z_y_dim, None)
    sigmaq_t1 = out_layer(hqz, z_y_dim, tf.nn.softplus)
    qzy = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
                 scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

    # end graph def

    # ------ Criticism / evaluation graph:
    zy_learned = ed.copy(qzy, {x: x_ph})

    # sample posterior predictive for p(y|z_y,t)
    y_post = ed.copy(y, {zy: qzy, t: t_ph}, scope='y_post')
    # crude approximation of the above
    y_post_mean = ed.copy(y, {zy: qzy.mean(), t: t_ph}, scope='y_post_mean')



    # ------ Training -  Run variational inference

    # Create data dictionary for edward
    data = {x: x_ph, y: y_ph, qt: t_ph, t: t_ph, qy: y_ph}

    batch_size = min(batch_size, n_train)
    n_iter_per_epoch = n_train // batch_size

    inference = ed.KLqp({zt: qzt, zy: qzy}, data=data)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    data_scaling = n_train / batch_size  # to scale likelihood againt prior
    inference.initialize(optimizer=optimizer, n_samples=5, n_iter=n_iter_per_epoch * n_epoch,
                         scale={x: data_scaling, t: data_scaling, y: data_scaling})


    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(n_epoch):
            train_generator = batch_generator(np.random.permutation(n_train), batch_size)
            avg_loss = 0.0
            for j in range(n_iter_per_epoch):
                # Take batch:
                idx = next(train_generator)
                x_b, t_b, y_b = x_train[idx], t_train[idx], y_train[idx]
                info_dict = inference.update(feed_dict={x_ph: x_b, t_ph: t_b, y_ph: y_b})
                inference.print_progress(info_dict)
                avg_loss += info_dict['loss']
            avg_loss = avg_loss / n_iter_per_epoch
            avg_loss = avg_loss / batch_size
            # print('Epoch {}, avg loss {}'.format(epoch, avg_loss))

        # ------ Evaluation -
        x_test = test_set['X']
        H_test = test_set['H']

        z_y_est = sess.run(zy_learned.mean(), feed_dict={x_ph: x_test})
        #
        # plt.scatter(x_test[:, 0].flatten(), z_y_est.flatten())
        # plt.xlabel('X_0')
        # plt.ylabel('z_y')
        # plt.show()

        z_y_test = sess.run(zy_learned.mean(), feed_dict={x_ph: x_test})
        z_y_train = sess.run(zy_learned.mean(), feed_dict={x_ph: x_train})

        # plt.scatter(x_test[:, 1].flatten(), z_y_test.flatten())
        # plt.xlabel('X_1')
        # plt.ylabel('z_y')
        # plt.show()
        #
        # plt.scatter(H_test.flatten(), z_y_test.flatten())
        # plt.xlabel('H')
        # plt.ylabel('z_y')
        # plt.show()

        # CATE estimation:
        if args.estimation_type == 'approx_posterior':
            forced_t = np.ones((args.n_test, 1))
            est_y0 = sess.run(y_post_mean.mean(), feed_dict={x_ph: x_test, t_ph: 0 * forced_t})
            est_y1 = sess.run(y_post_mean.mean(), feed_dict={x_ph: x_test, t_ph: forced_t})
            # std_y1 = sess.run(y_post.stddev(), feed_dict={x_ph: x_test, t_ph: forced_t})
        elif args.estimation_type == 'latent_matching':
            est_y0, est_y1 = matching_estimate(z_y_train, t_train, y_train, z_y_test)
        elif args.estimation_type == 'proxy_matching':
            est_y0, est_y1 = matching_estimate(x_train, t_train, y_train, x_test)
        else:
            raise ValueError('Unrecognised estimation_type')

        evalaute_effect_estimate(est_y0, est_y1, test_set,
                                 model_name='Separated CEVAE - Latent dims:  ' + str(latent_dims), estimation_type=args.estimation_type)
    # end session


