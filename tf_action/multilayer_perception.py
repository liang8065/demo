#encoding:utf-8
'''
Accuracy: 0.9526
'''
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 128
EPOCHS_NUM = 30

n_inputs = 784
n_classes = 10
n_hidden_1 = 256
n_hidden_2 = 256

# 数据
mnist = input_data.read_data_sets('../../data/mnist', one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

# placeholder
X = tf.placeholder(tf.float32, [BATCH_SIZE, n_inputs], name = "x_placeholder")
Y = tf.placeholder(tf.int32, [BATCH_SIZE, n_classes], name = "y_placeholder")

# 参数
weights = {
    "h1" : tf.Variable(tf.random_normal([n_inputs, n_hidden_1]), name='W1'),
    "h2" : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'out' : tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W')
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'out' : tf.Variable(tf.random_normal([n_classes]), name='bias')
}

# 目标
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'], name='fc_1')
    layer_1 = tf.nn.relu(layer_1, name='relu_1')
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'], name='fc_2')
    layer_2 = tf.nn.relu(layer_2, name='relu_2')
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name='fc_3')
    return out_layer

logits = multilayer_perceptron(X, weights, biases)

# loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(cross_entropy)

# 优化器
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./graphs/mlp_dnn", sess.graph)

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)
    for i in range(EPOCHS_NUM):
        total_loss = 0
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})
            total_loss += loss_batch
        print("Average loss epoch {0} : {1}".format(i, total_loss / n_batches))
    print("Total time: {0} seconds".format(time.time() - start_time))

    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    total_correct_preds = 0
    for _ in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
        accuracy_batch = sess.run([accuracy], feed_dict={X:X_batch, Y:Y_batch})
        total_correct_preds += accuracy_batch[0]
    
    print("Accuracy: {0}".format(total_correct_preds / mnist.test.num_examples))

    writer.close()