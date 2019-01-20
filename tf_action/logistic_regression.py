#encoding:utf-8
'''
Accuracy: 0.9147
'''
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 128
EPOCHS_NUM = 30

# 数据
mnist = input_data.read_data_sets('../../data/mnist', one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

# placeholder
X = tf.placeholder(tf.float32, [BATCH_SIZE, 784], name = "x_placeholder")
Y = tf.placeholder(tf.int32, [BATCH_SIZE, 10], name = "y_placeholder")

# 参数
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")

# 目标
logits = tf.matmul(X, w) + b

# loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(cross_entropy)

# 优化器
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./graphs/logistic_reg", sess.graph)

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