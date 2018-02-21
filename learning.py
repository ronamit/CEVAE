# ------- Learn Model -------------#

from __future__ import absolute_import, division, print_function
import numpy as np

import edward as ed
import matplotlib.pyplot as plt
import tensorflow as tf
from edward.models import Normal, Bernoulli

from utils import batch_generator


def learn_latent_model(args, x_train, t_train, y_train):
    if args.model_type == 'standard':
        learn_standard(args, x_train, t_train, y_train)
    elif args.model_type == 'separated':
        learn_separated(args, x_train, t_train, y_train)

#----------------------------------------------------------------------------------------#

def learn_standard(args, x_train, t_train, y_train):
    # Parameters
    n_hidd = 256  # number of hidden units
    n_epoch = args.n_epoch
    learning_rate = 0.001
    batch_size = 128


    n_samples = x_train.shape[0]
    x_dim = x_train.shape[1]

    batch_size = min(batch_size, n_samples)

    # ------ Define Graph ---------------------#

    # ------ Define Inputs ---------------------#
    # define placeholder which will receive data batches
    x_ph = tf.placeholder(tf.float32, [None, x_dim])
    t_ph = tf.placeholder(tf.float32, [None, 1])
    y_ph = tf.placeholder(tf.float32, [None, 1])

    # ------ Define generative model /decoder-----------------------#

    z_dim = 1

    # p(z) - prior over latent variables:
    z = Normal(loc=tf.zeros([batch_size, z_dim]), scale=tf.ones([batch_size, z_dim]))

    latent_dim = z_dim

    # p(x|z) - likelihood of proxy X
    hidden = tf.layers.dense(z, n_hidd, activation=tf.nn.relu)
    x = Normal(loc=tf.layers.dense(hidden, x_dim, activation=None),
               scale=tf.layers.dense(hidden, x_dim, activation=tf.nn.softplus),
               name='gaussian_px_z')

    # p(t|z)
    hidden = tf.layers.dense(z, n_hidd, activation=tf.nn.relu)
    probs = tf.layers.dense(hidden, 1, activation=tf.nn.sigmoid)  # output in [0,1]
    t = Bernoulli(probs=probs, dtype=tf.float32, name='bernoulli_pt_z')

    # p(y|t,z)
    hidden = tf.layers.dense(z, n_hidd, activation=tf.nn.relu)
    mu_y_t0 = tf.layers.dense(hidden, 1, activation=None)
    hidden = tf.layers.dense(z, n_hidd, activation=tf.nn.relu)
    mu_y_t1 = tf.layers.dense(hidden, 1, activation=None)
    # TODO: maybe share hidden
    y = Normal(loc=t * mu_y_t1 + (1. - t) * mu_y_t0, scale=tf.ones_like(mu_y_t0))
    # TODO: maybe learn variance pf y


    # ------ Define inference model - CEVAE variational approximation (encoder)

    # q(t|x)
    hqt = tf.layers.dense(x_ph, n_hidd, activation=tf.nn.relu)
    probs = tf.layers.dense(hqt, 1, activation=tf.nn.sigmoid)  # output in [0,1]
    qt = Bernoulli(probs=probs, dtype=tf.float32)

    # q(y|x,t)
    hqy = tf.layers.dense(x_ph, n_hidd, activation=tf.nn.relu)  # shared hidden layer
    mu_qy_t0 = tf.layers.dense(hqy, 1, activation=None)
    mu_qy_t1 = tf.layers.dense(hqy, 1, activation=tf.nn.relu)
    qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))

    # q(z|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = tf.layers.dense(inpt2, n_hidd, activation=tf.nn.relu) # shared hidden layer
    muq_t0 = tf.layers.dense(hqz, latent_dim, activation=None)
    sigmaq_t0 = tf.layers.dense(hqz, latent_dim, activation=tf.nn.softplus)
    muq_t1 = tf.layers.dense(hqz, latent_dim, activation=None)
    sigmaq_t1 = tf.layers.dense(hqz, latent_dim, activation=tf.nn.softplus)
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

    batch_size = min(batch_size, n_samples)
    n_iter_per_epoch = n_samples // batch_size

    inference = ed.KLqp({z: qz}, data=data)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    data_scaling = n_samples / batch_size  # to scale likelihood againt prior
    inference.initialize(optimizer=optimizer, n_samples=5, n_iter=n_iter_per_epoch * n_epoch,
                         scale={x: data_scaling, t: data_scaling, y: data_scaling})

    # end graph def

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(n_epoch):
            train_generator = batch_generator(np.random.permutation(n_samples), batch_size)
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
        z_est = sess.run(z_learned.mean(), feed_dict={x_ph: x_train})
        plt.scatter(x_train[:, 0].flatten(), z_est.flatten())
        plt.xlabel('X_0')
        plt.ylabel('Z')
        plt.show()

        z_est = sess.run(z_learned.mean(), feed_dict={x_ph: x_train})
        plt.scatter(x_train[:, 1].flatten(), z_est.flatten())
        plt.xlabel('X_1')
        plt.ylabel('Z')
        plt.show()

    # end session

#----------------------------------------------------------------------------------------------------------------------------#
def learn_separated(args, x_train, t_train, y_train):
    # Parameters
    n_hidd = 256  # number of hidden units
    n_epoch = args.n_epoch
    learning_rate = 0.001
    batch_size = 128

    n_samples = x_train.shape[0]
    x_dim = x_train.shape[1]

    batch_size = min(batch_size, n_samples)

    # ------ Define Graph ---------------------#

    # ------ Define Inputs ---------------------#
    # define placeholder which will receive data batches
    x_ph = tf.placeholder(tf.float32, [None, x_dim])
    t_ph = tf.placeholder(tf.float32, [None, 1])
    y_ph = tf.placeholder(tf.float32, [None, 1])

    # ------ Define generative model /decoder-----------------------#

    z_x_dim = 1
    z_t_dim = 1
    z_y_dim = 1
    # prior over latent variables:
    # p(zx) -
    zx = Normal(loc=tf.zeros([batch_size, z_x_dim]), scale=tf.ones([batch_size, z_x_dim]))
    # p(zt) -
    zt = Normal(loc=tf.zeros([batch_size, z_t_dim]), scale=tf.ones([batch_size, z_t_dim]))
    # p(zy) -
    zy = Normal(loc=tf.zeros([batch_size, z_y_dim]), scale=tf.ones([batch_size, z_y_dim]))


    # p(x|z) - likelihood of proxy X
    z = tf.concat([zx, zt, zy], axis=1)
    hidden = tf.layers.dense(z, n_hidd, activation=tf.nn.relu)
    x = Normal(loc=tf.layers.dense(hidden, x_dim, activation=None),
               scale=tf.layers.dense(hidden, x_dim, activation=tf.nn.softplus),
               name='gaussian_px_z')

    # p(t|zt)
    hidden = tf.layers.dense(zt, n_hidd, activation=tf.nn.relu)
    probs = tf.layers.dense(hidden, 1, activation=tf.nn.sigmoid)  # output in [0,1]
    t = Bernoulli(probs=probs, dtype=tf.float32, name='bernoulli_pt_z')

    # p(y|t,zy)
    hidden = tf.layers.dense(zy, n_hidd, activation=tf.nn.relu)
    mu_y_t0 = tf.layers.dense(hidden, 1, activation=None)
    hidden = tf.layers.dense(zy, n_hidd, activation=tf.nn.relu)
    mu_y_t1 = tf.layers.dense(hidden, 1, activation=None)
    # TODO: maybe share hidden
    y = Normal(loc=t * mu_y_t1 + (1. - t) * mu_y_t0, scale=tf.ones_like(mu_y_t0))
    # TODO: maybe learn variance pf y

    # ------ Define inference model - CEVAE variational approximation (encoder)

    # q(t|x)
    hqt = tf.layers.dense(x_ph, n_hidd, activation=tf.nn.relu)
    probs = tf.layers.dense(hqt, 1, activation=tf.nn.sigmoid)  # output in [0,1]
    qt = Bernoulli(probs=probs, dtype=tf.float32)

    # q(y|x,t)
    hqy = tf.layers.dense(x_ph, n_hidd, activation=tf.nn.relu)  # shared hidden layer
    mu_qy_t0 = tf.layers.dense(hqy, 1, activation=None)
    mu_qy_t1 = tf.layers.dense(hqy, 1, activation=tf.nn.relu)
    qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))

    # q(z_x|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = tf.layers.dense(inpt2, n_hidd, activation=tf.nn.relu)  # shared hidden layer
    muq_t0 = tf.layers.dense(hqz, z_x_dim, activation=None)
    sigmaq_t0 = tf.layers.dense(hqz, z_x_dim, activation=tf.nn.softplus)
    muq_t1 = tf.layers.dense(hqz, z_x_dim, activation=None)
    sigmaq_t1 = tf.layers.dense(hqz, z_x_dim, activation=tf.nn.softplus)
    qzx = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
                scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

    # q(z_t|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = tf.layers.dense(inpt2, n_hidd, activation=tf.nn.relu)  # shared hidden layer
    muq_t0 = tf.layers.dense(hqz, z_t_dim, activation=None)
    sigmaq_t0 = tf.layers.dense(hqz, z_t_dim, activation=tf.nn.softplus)
    muq_t1 = tf.layers.dense(hqz, z_t_dim, activation=None)
    sigmaq_t1 = tf.layers.dense(hqz, z_t_dim, activation=tf.nn.softplus)
    qzt = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
                scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

    # q(z_y|x,t,y)
    inpt2 = tf.concat([x_ph, qy], axis=1)
    hqz = tf.layers.dense(inpt2, n_hidd, activation=tf.nn.relu)  # shared hidden layer
    muq_t0 = tf.layers.dense(hqz, z_y_dim, activation=None)
    sigmaq_t0 = tf.layers.dense(hqz, z_y_dim, activation=tf.nn.softplus)
    muq_t1 = tf.layers.dense(hqz, z_y_dim, activation=None)
    sigmaq_t1 = tf.layers.dense(hqz, z_y_dim, activation=tf.nn.softplus)
    qzy = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
                 scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

    # ------ Criticism / evaluation graph:
    zy_learned = ed.copy(qzy, {x: x_ph})

    # sample posterior predictive for p(y|z_y,t)
    y_post = ed.copy(y, {zy: qzy, t: t_ph}, scope='y_post')
    # crude approximation of the above
    y_post_mean = ed.copy(y, {zy: qzy.mean(), t: t_ph}, scope='y_post_mean')

    # ------ Training -  Run variational inference

    # Create data dictionary for edward
    data = {x: x_ph, y: y_ph, qt: t_ph, t: t_ph, qy: y_ph}

    batch_size = min(batch_size, n_samples)
    n_iter_per_epoch = n_samples // batch_size

    inference = ed.KLqp({zx: qzx, zt: qzt, zy: qzy}, data=data)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    data_scaling = n_samples / batch_size  # to scale likelihood againt prior
    inference.initialize(optimizer=optimizer, n_samples=5, n_iter=n_iter_per_epoch * n_epoch,
                         scale={x: data_scaling, t: data_scaling, y: data_scaling})

    # end graph def

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(n_epoch):
            train_generator = batch_generator(np.random.permutation(n_samples), batch_size)
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
        z_y_est = sess.run(zy_learned.mean(), feed_dict={x_ph: x_train})

        plt.scatter(x_train[:, 0].flatten(), z_y_est.flatten())
        plt.xlabel('X_0')
        plt.ylabel('z_y')
        plt.show()

        z_y_est = sess.run(zy_learned.mean(), feed_dict={x_ph: x_train})
        plt.scatter(x_train[:, 1].flatten(), z_y_est.flatten())
        plt.xlabel('X_1')
        plt.ylabel('z_y')
        plt.show()

