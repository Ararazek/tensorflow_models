import tensorflow as tf
import numpy as np
import datetime

from keras.datasets import mnist as mnist_object

def to_onehot(nclasses, index):
    aux = np.zeros((nclasses))
    aux[index] = 1
    return aux

num_classes = 10
learning_rate = 0.000008
epochs = 50
batch_size = 20

(train_x, aux), (test_x, aux2) = mnist_object.load_data()
train_y = [ to_onehot(num_classes, i) for i in aux ]
test_y = [ to_onehot(num_classes, i) for i in aux2 ]
train_x = train_x/255
test_x = test_x/255

num_samples = len(train_x)

num_gpus = 2

split_batch_size = int(batch_size/num_gpus)

def to_onehot(nclasses, index):
    aux = np.zeros((nclasses))
    aux[index] = 1
    return aux

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def model(inp, reuse, dropout_rate):
    with tf.variable_scope('model_net', reuse=reuse):

        x = tf.layers.flatten(inp)

        x = tf.layers.dense(
            inputs = x,
            units = 256,
            activation = tf.nn.relu
        )

        out = tf.layers.dense(
            inputs = x,
            units = 10,
            activation = tf.nn.relu
        )

        return out


with tf.device('/cpu:0'):

    tower_grads = []
    reuse = False

    x_ = tf.placeholder(tf.float32, [None, 28, 28])
    y = tf.placeholder(tf.float32, [None, 10])

    #dr = tf.placeholder(tf.float32, ())

    for i in range(num_gpus):
        with tf.device("/gpu:{}".format(i)):

            # Split data between GPUs
            subx = x_[i*split_batch_size:(i+1)*split_batch_size]
            suby = y[i*split_batch_size:(i+1)*split_batch_size]

            logits = model(subx, reuse, 0.3)

            test_logits = model(x_, True, 1)

            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=suby))
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            grads = optimizer.compute_gradients(loss_op)
            tower_grads.append(grads)

            reuse = True

            if i == 0:
                softmax = tf.layers.dense(inputs=test_logits, units=10, activation=tf.nn.softmax)
                correct_pred = tf.equal(tf.argmax(test_logits, 1), tf.argmax(y, 1))
                accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    gradient = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(gradient)


    with tf.Session() as sess:

        var_init = tf.global_variables_initializer()

        (train_x, aux), (test_x, aux2) = mnist_object.load_data()
        train_y = [ to_onehot(num_classes, i) for i in aux ]
        test_y = [ to_onehot(num_classes, i) for i in aux2 ]
        train_x = train_x/255
        test_x = test_x/255

        sess.run(var_init)

        for epoch in range(epochs):
            t_ini = datetime.datetime.now()
            for nbatch in range(int(num_samples/batch_size)-1):
                index = nbatch*batch_size
                xb = train_x[index:index+batch_size]
                yb = train_y[index:index+batch_size]
                sess.run(train_op, {x_: xb, y: yb})
                if nbatch % 10 == 0:
                    loss, acc = sess.run(
                        [loss_op, accuracy_op],
                        {x_: test_x, y: test_y}
                    )
                    t_now = datetime.datetime.now() - t_ini
                    print('Epoch -> {0:d}, train loss -> {1:.3f}, validation accuracy -> {2:.3f}, time this epoch -> {3:.3f}'.format(epoch, loss, acc, t_now.total_seconds()), end='\r')
            print("")
