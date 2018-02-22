# ------- Learn Model -------------#

from __future__ import absolute_import, division, print_function
import numpy as np

import edward as ed
import matplotlib.pyplot as plt
import tensorflow as tf
from edward.models import Normal, Bernoulli

from Utils import batch_generator, evalaute_effect_estimate, get_fc_layer_fn

#----------------------------------------------------------------------------------------#
def learn_latent_model(args, train_set, test_set):

    n_train = args.n_train



    if args.model_type == 'standard':
        learn_standard(args, train_set, test_set)

    elif args.model_type == 'separated':
        learn_separated(args, train_set, test_set)

    elif args.model_type == 'supervised':
        learn_supervised(args, train_set, test_set)

    else:
        raise ValueError('Unrecognized model_type')
        
#----------------------------------------------------------------------------------------#

def learn_standard(args, train_set, test_set):

    # Parameters
    n_hidd = 1024  # number of hidden units
    n_epoch = args.n_epoch
    learning_rate = 0.001
    batch_size = 128

    fc_layer = get_fc_layer_fn(l2_reg_scale=1e-4)

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

    z_dim = 1

    # p(z) - prior over latent variables:
    z = Normal(loc=tf.zeros([n_ph, z_dim]), scale=tf.ones([n_ph, z_dim]))

    latent_dim = z_dim

    # p(x|z) - likelihood of proxy X
    hidden = fc_layer(z, n_hidd, tf.nn.sigmoid)
    x = Normal(loc=fc_layer(hidden, x_dim, None,),
               scale=fc_layer(hidden, x_dim, tf.nn.softplus),
               name='gaussian_px_z')

    # p(t|z)
    hidden = fc_layer(z, n_hidd, tf.nn.sigmoid)
    probs = fc_layer(hidden, 1, tf.nn.sigmoid)  # output in [0,1]
    t = Bernoulli(probs=probs, dtype=tf.float32, name='bernoulli_pt_z')

    # p(y|t,z)
    hidden = fc_layer(z, n_hidd, tf.nn.elu)
    mu_y_t0 = fc_layer(hidden, 1, None)
    hidden = fc_layer(z, n_hidd, tf.nn.elu)
    mu_y_t1 = fc_layer(hidden, 1, None)
    # TODO: maybe share hidden
    y = Normal(loc=t * mu_y_t1 + (1. - t) * mu_y_t0, scale=tf.ones_like(mu_y_t0))
    # TODO: maybe learn variance pf y


    # ------ Define inference model - CEVAE variational approximation (encoder)

    # q(t|x)
    hqt = fc_layer(x_ph, n_hidd, tf.nn.elu)
    probs = fc_layer(hqt, 1, tf.nn.sigmoid)  # output in [0,1]
    qt = Bernoulli(probs=probs, dtype=tf.float32)

    # q(y|x,t)
    hqy = fc_layer(x_ph, n_hidd, tf.nn.elu)  # shared hidden layer
    mu_qy_t0 = fc_layer(hqy, 1, None)
    mu_qy_t1 = fc_layer(hqy, 1, tf.nn.elu)
    qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))

    # q(z|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = fc_layer(inpt2, n_hidd, tf.nn.elu) # shared hidden layer
    muq_t0 = fc_layer(hqz, latent_dim, None)
    sigmaq_t0 = fc_layer(hqz, latent_dim, tf.nn.softplus)
    muq_t1 = fc_layer(hqz, latent_dim, None)
    sigmaq_t1 = fc_layer(hqz, latent_dim, tf.nn.softplus)
    qz = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
                scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

    # ------ Criticism / evaluation graph:
    z_learned = ed.copy(qz, {x: x_ph})

    # sample posterior predictive for p(y|z,t)
    y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
    # crude approximation of the above
    y_post_mean = ed.copy(y, {z: qz.mean(), t: t_ph}, scope='y_post_mean')

    # ------ Training -  Run variational inference

    # Create data dictionary for edward
    data = {x: x_ph, y: y_ph, qt: t_ph, t: t_ph, qy: y_ph}

    batch_size = min(batch_size, n_train)
    n_iter_per_epoch = n_train // batch_size

    inference = ed.KLqp({z: qz}, data=data)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    data_scaling = n_train / batch_size  # to scale likelihood againt prior
    inference.initialize(optimizer=optimizer, n_samples=5, n_iter=n_iter_per_epoch * n_epoch,
                         scale={x: data_scaling, t: data_scaling, y: data_scaling})


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
                info_dict = inference.update(feed_dict={x_ph: x_b, t_ph: t_b, y_ph: y_b})
                inference.print_progress(info_dict)
                avg_loss += info_dict['loss']
            avg_loss = avg_loss / n_iter_per_epoch
            avg_loss = avg_loss / batch_size
            # print('Epoch {}, avg loss {}'.format(epoch, avg_loss))



        # ------ Evaluation -
        x_test = test_set['X']

        z_est = sess.run(z_learned.mean(), feed_dict={x_ph: x_test})
        plt.scatter(x_test[:, 0].flatten(), z_est.flatten())
        plt.xlabel('X_0')
        plt.ylabel('Z')
        plt.show()

        z_est = sess.run(z_learned.mean(), feed_dict={x_ph: x_test})
        plt.scatter(x_test[:, 1].flatten(), z_est.flatten())
        plt.xlabel('X_1')
        plt.ylabel('Z')
        plt.show()

        # CATE estimation:
        forced_t = np.ones((args.n_test, 1))
        est_y1 = sess.run(y_post.mean(), feed_dict={x_ph: x_test, t_ph: forced_t})
        est_y0 = sess.run(y_post.mean(), feed_dict={x_ph: x_test, t_ph: 0*forced_t})
    # end session

    evalaute_effect_estimate(est_y0, est_y1, test_set, method_name='Posterior - Standard')

#----------------------------------------------------------------------------------------------------------------------------#
def learn_separated(args, train_set, test_set):

    # Parameters
    n_hidd = 1024  # number of hidden units
    n_epoch = args.n_epoch
    learning_rate = 0.001
    batch_size = 128

    fc_layer = get_fc_layer_fn(l2_reg_scale=1e-4)

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

    z_x_dim = 1
    z_t_dim = 1
    z_y_dim = 1
    # prior over latent variables:
    # p(zx) -
    zx = Normal(loc=tf.zeros([n_ph, z_x_dim]), scale=tf.ones([n_ph, z_x_dim]))
    # p(zt) -
    zt = Normal(loc=tf.zeros([n_ph, z_t_dim]), scale=tf.ones([n_ph, z_t_dim]))
    # p(zy) -
    zy = Normal(loc=tf.zeros([n_ph, z_y_dim]), scale=tf.ones([n_ph, z_y_dim]))


    # p(x|z) - likelihood of proxy X
    z = tf.concat([zx, zt, zy], axis=1)
    hidden = fc_layer(z, n_hidd, tf.nn.elu)
    x = Normal(loc=fc_layer(hidden, x_dim, None),
               scale=fc_layer(hidden, x_dim, tf.nn.softplus),
               name='gaussian_px_z')

    # p(t|zt)
    hidden = fc_layer(zt, n_hidd, tf.nn.elu)
    probs = fc_layer(hidden, 1, tf.nn.sigmoid)  # output in [0,1]
    t = Bernoulli(probs=probs, dtype=tf.float32, name='bernoulli_pt_z')

    # p(y|t,zy)
    hidden = fc_layer(zy, n_hidd, tf.nn.elu)
    mu_y_t0 = fc_layer(hidden, 1, None)
    hidden = fc_layer(zy, n_hidd, tf.nn.elu)
    mu_y_t1 = fc_layer(hidden, 1, None)
    # TODO: maybe share hidden
    y = Normal(loc=t * mu_y_t1 + (1. - t) * mu_y_t0, scale=tf.ones_like(mu_y_t0))
    # TODO: maybe learn variance pf y

    # ------ Define inference model - CEVAE variational approximation (encoder)

    # q(t|x)
    hqt = fc_layer(x_ph, n_hidd, tf.nn.elu)
    probs = fc_layer(hqt, 1, tf.nn.sigmoid)  # output in [0,1]
    qt = Bernoulli(probs=probs, dtype=tf.float32)

    # q(y|x,t)
    hqy = fc_layer(x_ph, n_hidd, tf.nn.elu)  # shared hidden layer
    mu_qy_t0 = fc_layer(hqy, 1, None)
    mu_qy_t1 = fc_layer(hqy, 1, tf.nn.elu)
    qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))

    # q(z_x|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = fc_layer(inpt2, n_hidd, tf.nn.elu)  # shared hidden layer
    muq_t0 = fc_layer(hqz, z_x_dim, None)
    sigmaq_t0 = fc_layer(hqz, z_x_dim, tf.nn.softplus)
    muq_t1 = fc_layer(hqz, z_x_dim, None)
    sigmaq_t1 = fc_layer(hqz, z_x_dim, tf.nn.softplus)
    qzx = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
                scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

    # q(z_t|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = fc_layer(inpt2, n_hidd, tf.nn.elu)  # shared hidden layer
    muq_t0 = fc_layer(hqz, z_t_dim, None)
    sigmaq_t0 = fc_layer(hqz, z_t_dim, tf.nn.softplus)
    muq_t1 = fc_layer(hqz, z_t_dim, None)
    sigmaq_t1 = fc_layer(hqz, z_t_dim, tf.nn.softplus)
    qzt = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
                scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

    # q(z_y|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = fc_layer(inpt2, n_hidd, tf.nn.elu)  # shared hidden layer
    muq_t0 = fc_layer(hqz, z_y_dim, None)
    sigmaq_t0 = fc_layer(hqz, z_y_dim, tf.nn.softplus)
    muq_t1 = fc_layer(hqz, z_y_dim, None)
    sigmaq_t1 = fc_layer(hqz, z_y_dim, tf.nn.softplus)
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

    inference = ed.KLqp({zx: qzx, zt: qzt, zy: qzy}, data=data)
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

        z_y_est = sess.run(zy_learned.mean(), feed_dict={x_ph: x_test})

        plt.scatter(x_test[:, 0].flatten(), z_y_est.flatten())
        plt.xlabel('X_0')
        plt.ylabel('z_y')
        plt.show()

        z_y_est = sess.run(zy_learned.mean(), feed_dict={x_ph: x_test})
        plt.scatter(x_test[:, 1].flatten(), z_y_est.flatten())
        plt.xlabel('X_1')
        plt.ylabel('z_y')
        plt.show()

        # CATE estimation:
        forced_t = np.ones((args.n_test, 1))
        est_y1 = sess.run(y_post.mean(), feed_dict={x_ph: x_test, t_ph: forced_t})
        est_y0 = sess.run(y_post.mean(), feed_dict={x_ph: x_test, t_ph: 0*forced_t})
    # end session

    evalaute_effect_estimate(est_y0, est_y1, test_set, method_name='Posterior - Separated Latents')

#----------------------------------------------------------------------------------------------------------------------------#
def learn_supervised(args, train_set, test_set):

    # Parameters
    n_hidd = 1024  # number of hidden units
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
    # end session

    evalaute_effect_estimate(est_y0, est_y1, test_set, method_name='supervised')
