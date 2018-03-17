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
include_tensorboard = True
tensorboard_folder = "simple"

(train_x, aux), (test_x, aux2) = mnist_object.load_data()
train_y = [ to_onehot(num_classes, i) for i in aux ]
test_y = [ to_onehot(num_classes, i) for i in aux2 ]
train_x = train_x/255
test_x = test_x/255

num_samples = len(train_x)

# model
x_ = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.float32, [None, 10])

x = tf.layers.flatten(x_)
x = tf.layers.dense(inputs = x, units = 2048, activation = tf.nn.relu)
out = tf.layers.dense(inputs = x, units = 10, activation = tf.nn.relu)

loss_op = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('train loss', loss_op)
tf.summary.scalar('validation accuracy', accuracy_op)

sess = tf.Session()
var_init = tf.global_variables_initializer()
sess.run(var_init)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(tensorboard_folder, sess.graph)

for epoch in range(epochs):
    t_ini = datetime.datetime.now()
    summary = None
    for nbatch in range(int(num_samples/batch_size)-1):
        index = nbatch*batch_size
        xb = train_x[index:index+batch_size]
        yb = train_y[index:index+batch_size]
        sess.run(train_op, {x_: xb, y: yb})
        if nbatch % 10 == 0:
            if include_tensorboard:
                loss, acc, summary = sess.run(
                    [loss_op, accuracy_op, merged],
                    {x_: test_x, y: test_y}
                )
            else:
                loss, acc = sess.run(
                    [loss_op, accuracy_op],
                    {x_: test_x, y: test_y}
                )
            t_now = datetime.datetime.now() - t_ini
            print('Epoch -> {0:d}, train loss -> {1:.3f}, validation accuracy -> {2:.3f}, epoch time (secs) -> {3:.3f}s'.format(epoch, loss, acc, t_now.total_seconds()), end='\r')
        if include_tensorboard:
            writer.add_summary(summary, epoch)
    print("")
