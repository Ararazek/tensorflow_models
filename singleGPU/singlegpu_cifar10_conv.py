import tensorflow as tf
import numpy as np
import datetime

from keras.datasets import cifar10 as cifar10_object

def to_onehot(nclasses, index):
    aux = np.zeros((nclasses))
    aux[index] = 1
    return aux

num_classes = 10
learning_rate = 0.000015
epochs = 200
batch_size = 20
dropout_rate = 0.4
include_tensorboard = True
tensorboard_folder = "cifar10conv"

(train_x, aux), (test_x, aux2) = cifar10_object.load_data()
train_y = [ to_onehot(num_classes, i) for i in aux ]
test_y = [ to_onehot(num_classes, i) for i in aux2 ]
train_x = train_x/255
test_x = test_x/255

num_samples = len(train_x)
nbatches = int(num_samples/batch_size)

# model
x_ = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

drop_rate = tf.placeholder(tf.float32)

x = tf.layers.conv2d(
    inputs = x_,
    filters = 32,
    kernel_size = [5, 5],
    padding = 'same',
    activation = tf.nn.elu
)

x = tf.layers.batch_normalization(x)

x = tf.layers.conv2d(
    inputs = x,
    filters = 32,
    kernel_size = [5, 5],
    padding = 'same',
    activation = tf.nn.elu
)

x = tf.layers.batch_normalization(x)

x = tf.layers.max_pooling2d(
    inputs = x,
    pool_size = [2, 2],
    strides = 2
)

x = tf.layers.dropout(x, rate=drop_rate, training = tf.not_equal(drop_rate, 1))

x = tf.layers.conv2d(
    inputs = x,
    filters = 64,
    kernel_size = [3, 3],
    padding = 'same',
    activation = tf.nn.elu
)

x = tf.layers.batch_normalization(x)

x = tf.layers.conv2d(
    inputs = x,
    filters = 64,
    kernel_size = [3, 3],
    padding = 'same',
    activation = tf.nn.elu
)

x = tf.layers.batch_normalization(x)

x = tf.layers.max_pooling2d(
    inputs = x,
    pool_size = [2, 2],
    strides = 2
)

x = tf.layers.dropout(x, rate=drop_rate, training = tf.not_equal(drop_rate, 1))

x = tf.layers.conv2d(
    inputs = x,
    filters = 96,
    kernel_size = [3, 3],
    padding = 'same',
    activation = tf.nn.elu
)

x = tf.layers.batch_normalization(x)

x = tf.layers.conv2d(
    inputs = x,
    filters = 96,
    kernel_size = [3, 3],
    padding = 'same',
    activation = tf.nn.elu
)

x = tf.layers.batch_normalization(x)

x = tf.layers.max_pooling2d(
    inputs = x,
    pool_size = [2, 2],
    strides = 2
)

x = tf.layers.dropout(x, rate=drop_rate, training = tf.not_equal(drop_rate, 1))

flat = tf.layers.flatten(x)

dense1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.elu)

out = tf.layers.dense(inputs=dense1, units=10, activation=tf.nn.elu)

loss_op = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

val_loss = tf.summary.scalar('validation loss', loss_op)
val_acc = tf.summary.scalar('validation accuracy', accuracy_op)

train_loss = tf.summary.scalar('train loss', loss_op)
train_acc = tf.summary.scalar('train accuracy', accuracy_op)

sess = tf.Session()
var_init = tf.global_variables_initializer()
sess.run(var_init)

merged_validation = tf.summary.merge([val_loss, val_acc])
merged_train = tf.summary.merge([train_loss, train_acc])
writer = tf.summary.FileWriter(tensorboard_folder, sess.graph)

for epoch in range(epochs):
    t_ini = datetime.datetime.now()
    summary = None
    acc = 0
    loss = 0
    train_acc = 0
    train_loss = 0

    for nbatch in range(nbatches-1):
        index = nbatch*batch_size
        xb = train_x[index:index+batch_size]
        yb = train_y[index:index+batch_size]
        sess.run(train_op, {x_: xb, y: yb, drop_rate: dropout_rate})

        percentile = (nbatch*100)/nbatches
        print("Epoch {4:3d}: {0:.2f}{1} of training samples [{2:5d}/{3:5d}]".format(percentile, '%', nbatch*batch_size, nbatches*batch_size, epoch), end="\r")
    if include_tensorboard:
        loss, acc, summary = sess.run(
            [loss_op, accuracy_op, merged_validation],
            {x_: test_x, y: test_y, drop_rate: 1}
        )
        writer.add_summary(summary, epoch)
        train_loss, train_acc, summary = sess.run(
            [loss_op, accuracy_op, merged_train],
            {x_: train_x[0:10000], y: train_y[0:10000], drop_rate: 1}
        )
        writer.add_summary(summary, epoch)
    else:
        loss, acc = sess.run(
            [loss_op, accuracy_op],
            {x_: test_x, y: test_y, drop_rate: 1}
        )
    t_now = datetime.datetime.now() - t_ini
    print('Epoch {0:3d}: validation loss/acc -> {1:.3f}/{2:.3f}, train loss/acc -> {3:.3f}/{4:.3f}, epoch time (secs) -> {5:.3f}s'.format(epoch, loss, acc, train_loss, train_acc, t_now.total_seconds()))
