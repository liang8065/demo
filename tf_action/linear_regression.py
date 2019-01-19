#encoding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 数据
n_observations = 100
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
'''
plt.scatter(xs, ys)
plt.show()
'''

# placeholder
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 参数
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 目标
Y_pred = tf.add(tf.multiply(X, W), b)

# loss
loss = tf.square(Y - Y_pred, name='loss')

# 优化器
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# training
n_samples = xs.shape[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    for i in range(50):
        total_loss = 0
        for x, y in zip(xs, ys):
            _, l = sess.run([optimizer, loss], feed_dict={X : x, Y : y})
            total_loss += l
        if i % 5 == 0:
            print("Epoch {0} ：{1}".format(i, total_loss / n_samples))
    
    writer.close()

    dst_W, dst_b = sess.run([W, b])
    print("W = {0}, b = {1}".format(dst_W, dst_b))
    plt.plot(xs, ys, 'bo', label='Real data')
    plt.plot(xs, xs * dst_W + dst_b, 'r', label='Predicted data')
    plt.legend()
    plt.show()
