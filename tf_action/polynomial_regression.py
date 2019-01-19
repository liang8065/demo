#encoding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 数据
n_observations = 100
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)

# placeholder
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 参数
W1 = tf.Variable(tf.random_normal([1]), name='weight_1')
W2 = tf.Variable(tf.random_normal([1]), name='weight_2')
W3 = tf.Variable(tf.random_normal([1]), name='weight_3')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 目标
Y_pred = tf.add(tf.multiply(X, W1), b)
Y_pred = tf.add(tf.multiply(tf.pow(X, 2), W2), Y_pred)
Y_pred = tf.add(tf.multiply(tf.pow(X, 3), W3), Y_pred)

# loss
sample_num = xs.shape[0]
loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / sample_num

# 优化器
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/polynomial_reg', sess.graph)

    for i in range(50):
        total_loss = 0
        for x, y in zip(xs, ys):
            _, l = sess.run([optimizer, loss], feed_dict={X : x, Y : y})
            total_loss += l
        if i % 5 == 0:
            print("Epoch {0} ：{1}".format(i, total_loss / sample_num))
    
    writer.close()

    dst_W1, dst_W2, dst_W3, dst_b = sess.run([W1, W2, W3, b])
    print("W1 = {0}, W2={1}, W3={2}, b = {3}".format(dst_W1, dst_W2, dst_W3, dst_b))
    y_predict = xs * dst_W1 + xs * np.power(xs, 2) * dst_W2 + np.power(xs, 3) * dst_W3 + dst_b
    plt.plot(xs, ys, 'bo', label='Real data')
    plt.plot(xs, y_predict, 'r', label='Predicted data')
    plt.legend()
    plt.show()