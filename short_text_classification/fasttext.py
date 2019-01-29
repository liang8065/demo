#encoding: utf-8
import sys
import os
import pickle
import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("all_data_h5py","../../data/short_text_classification/data_to_id.h5","all training data")
tf.app.flags.DEFINE_string("id_index_pkl","../../data/short_text_classification/vocabulary_dict.pkl","word to id, and type to id")
tf.app.flags.DEFINE_string("ckpt_dir","../../data/short_text_classification/fast_text_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_string("summary_dir","../../data/short_text_classification/summary_dir/","summary log path")
tf.app.flags.DEFINE_integer("label_size", 20, "number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.8, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_integer("num_sampled", 20, "number of noise sampling")  # 必须小于真实label数
tf.app.flags.DEFINE_integer("sentence_len",20,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",2,"epoch numbers")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every epochs.") 
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")


class fastText:
    def __init__(self, label_size, learning_rate, batch_size, decay_steps, decay_rate, num_sampled, sentence_len, vocab_size, embed_size, is_training):
        self.label_size = label_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.num_sampled = num_sampled
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training

        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")    # [batch_size, sentence_len]
        self.labels = tf.placeholder(tf.int32, [None], name="Labels")   # [batch_size]

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False,name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.instantiate_weights()

        self.logits = self.inference()  # [batch_size, label_size]

        #if not is_training:
        #    return

        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def instantiate_weights(self):
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size])
        self.W = tf.get_variable("W", [self.embed_size, self.label_size])
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.sentence) # [batch_size, sentence_len, embed_size]
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)  # [batch_size, embed_size]
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b    # [batch_size, label_size]
        return logits

    def loss(self, l2_lambda=0.01):
        labels = tf.reshape(self.labels, [-1])
        labels = tf.expand_dims(labels,1)   # [batch_size, 1]
        loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.W),    # [label_size, embed_size]
                               biases=self.b,   # [embed_size]
                               labels=labels,   # [batch_size, 1]
                               inputs=self.sentence_embeddings, # [batch_size, embed_size]
                               num_sampled=self.num_sampled,
                               num_classes=self.label_size,
                               partition_strategy="div"))
        #l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam")
        return train_op


def load_data(cache_file_h5py, cache_file_pickle):
    f_data = h5py.File(cache_file_h5py, 'r')
    print("f_data.keys:",list(f_data.keys()))
    train_X = f_data['train_X']
    train_Y = f_data['train_Y']
    vaild_X = f_data['valid_X']
    valid_Y = f_data['valid_Y']
    test_X = f_data['test_X']
    test_Y = f_data['test_Y']
    print("train_X.shape:", train_X.shape)
    print("train_Y.shape:", train_Y.shape)
    print("vaild_X.shape:", vaild_X.shape)
    print("valid_Y.shape:", valid_Y.shape)
    print("test_X.shape:", test_X.shape)
    print("test_Y.shape:", test_Y.shape)

    word2index, label2index = None,None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index = pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    print("word2index.size: ", len(word2index))
    print("label2index.size: ", len(label2index))
    return word2index, label2index, train_X,train_Y,vaild_X,valid_Y,test_X,test_Y


def do_eval(sess,fast_text,evalX,evalY,batch_size):
    number_examples = len(evalX)
    eval_loss,eval_acc,eval_counter = 0.0, 0.0, 0
    for start,end in zip(range(0,number_examples,batch_size), range(batch_size,number_examples,batch_size)):
        curr_eval_loss, curr_eval_acc, = sess.run([fast_text.loss_val, fast_text.accuracy],
                                          feed_dict={fast_text.sentence: evalX[start:end],fast_text.labels: evalY[start:end]})
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)


def main(_):
    word2index, label2index, trainX, trainY, vaildX, validY, testX, testY = load_data(FLAGS.all_data_h5py, FLAGS.id_index_pkl)

    vocab_size = len(word2index)

    fast_text = fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,
                            FLAGS.num_sampled, FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.summary_dir, tf.get_default_graph())

        curr_epoch = sess.run(fast_text.epoch_step)
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])
                    print("trainY[start:end]:",trainY[start:end])
                curr_loss, curr_acc, _ = sess.run([fast_text.loss_val, fast_text.accuracy, fast_text.train_op],
                                                    feed_dict={fast_text.sentence : trainX[start:end],
                                                               fast_text.labels : trainY[start:end]})
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 500 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch, counter, loss/float(counter), acc/float(counter)))

            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)

            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, fast_text, vaildX, validY, batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))

                save_path = FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess, save_path, global_step=fast_text.epoch_step)

        test_loss, test_acc = do_eval(sess, fast_text, testX, testY, batch_size)
        print("Test Loss:%.3f\tTest Accuracy:%.3f" %(test_loss, test_acc))

        writer.close()


def predict():
    word2index, label2index, trainX, trainY, vaildX, validY, testX, testY = load_data(FLAGS.all_data_h5py, FLAGS.id_index_pkl)
    vocab_size = len(word2index)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        fast_text = fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,
                            FLAGS.num_sampled, FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)

        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Model Failed')
            return

        print("test_X.shape:", testX.shape)
        print("test_Y.shape:", testY.shape)

        raw_labels = []
        predicted_labels = []
        number_examples = len(testX)
        batch_size = FLAGS.batch_size
        for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
            predictions = sess.run(fast_text.predictions, feed_dict={fast_text.sentence: testX[start:end], fast_text.labels: testY[start:end]})
            if len(predictions) == len(testY[start:end]):
                raw_labels.extend(testY[start:end])
                predicted_labels.extend(predictions)

        print classification_report(raw_labels, predicted_labels)


if __name__ == "__main__":
    if FLAGS.is_training:
        tf.app.run()
    else:
        predict()

