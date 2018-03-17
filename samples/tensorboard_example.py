import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = mnist.train.next_batch(100)
        #k = FLAGS.dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x_: xs, y_: ys}

x_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

dense = tf.layers.dense(
    inputs = x_,
    units = 256,
    activation = tf.nn.relu
)

output = tf.layers.dense(
    inputs = dense,
    units = 10,
    activation = tf.nn.relu
)

loss = tf.nn.softmax_cross_entropy_with_logits(
    labels = y_,
    logits = output
)

step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Accuracy
correct_predictions = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Session definition
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Tensorboard
variable_summaries(loss)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('tb/train', sess.graph)
test_writer = tf.summary.FileWriter('tb/test')

epochs = 20

for i in range(epochs):
    for j in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(step, feed_dict={x_: batch_xs, y_: batch_ys})

        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)

    print('Accuracy @ epoch '+str(i)+' : '+str(sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels})))


sess.close()
