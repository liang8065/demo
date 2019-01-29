#encoding:utf-8
import tensorflow as tf
import numpy as np

class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate,
                decay_steps, decay_rate, sequence_length, vocab_size, embed_size,
                initializer = tf.random_normal_initializer(stddev=0.1),
                clip_gradients = 5.0):
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.initializer = initializer
        self.clip_gradients = clip_gradients

        self.num_filters_total = self.num_filters * len(filter_sizes)

        # 输入
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name = "input_y")
        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 参数
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False,name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.instantiate_weights()

        # 目标
        self.logits = self.inference()
        self.possibility = tf.nn.sigmoid(self.logits)
        self.loss_val = self.loss()

        # 优化
        self.train_op = self.train()

        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def instantiate_weights(self):
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                            initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])
    
    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x) #[None,sentence_length,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1) #[None,sentence_length,embed_size,1)
        h = self.cnn_single_layer()
        with tf.name_scope("output"):
            logits = tf.matmul(h, self.W_projection) + self.b_projection
        return logits
    
    def cnn_single_layer(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("convolution-pooling-%s" % filter_size):
                # [filter_height, filter_width, in_channels, out_channels]
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                initializer=self.initializer)
                # input: [batch_size, in_height, in_width, in_channels]
                # out: [batch_size, sequence_length - filter_size + 1, 1, num_filters]
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",name="conv")
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")
                # [batch_size, 1, 1, num_filters]
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1], padding='VALID',name="pool")
                pooled_outputs.append(pooled)
        
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)
        h = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)

        return h

    def loss(self,l2_lambda=0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss
    
    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                        self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                        learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op


def test():
    filter_sizes = [2,3,4]
    num_filters = 128
    num_classes = 5
    learning_rate = 0.001
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.95
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100

    text_cnn = TextCNN(filter_sizes, num_filters, num_classes, learning_rate, decay_steps, decay_rate, sequence_length, vocab_size, embed_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1):
            input_x = np.random.randn(batch_size,sequence_length)
            input_x[input_x > 0] = 1
            input_x[input_x < 0] = 0
            input_y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
            print input_x
            print input_y
            loss,possibility,_=sess.run([text_cnn.loss_val,text_cnn.possibility,text_cnn.train_op],
                                        feed_dict={text_cnn.input_x:input_x,text_cnn.input_y:input_y,
                                                   text_cnn.dropout_keep_prob:1.0,text_cnn.is_training_flag : True})
            print(i,"loss:",loss)
            print("possibility:",possibility)


#test()