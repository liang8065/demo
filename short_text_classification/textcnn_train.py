#encoding: utf-8
import sys
import os
import pickle
import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from textcnn_model import TextCNN

reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("all_data_h5py","../../data/short_text_classification/data_to_id.h5","all training data")
tf.app.flags.DEFINE_string("id_index_pkl","../../data/short_text_classification/vocabulary_dict.pkl","word to id, and type to id")
tf.app.flags.DEFINE_string("ckpt_dir","../../data/short_text_classification/text_cnn_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_string("summary_dir","../../data/short_text_classification/summary_dir/","summary log path")
tf.app.flags.DEFINE_integer("label_size", 20, "number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_integer("sentence_len", 20, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 5, "epoch numbers")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every epochs.") 
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
filter_sizes=[2,3,4]


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


def do_eval(sess, text_cnn, evalX, evalY, batch_size):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start,end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        curr_eval_loss, curr_eval_acc, = sess.run([text_cnn.loss_val, text_cnn.accuracy],
                                          feed_dict={text_cnn.input_x: evalX[start:end],
                                                     text_cnn.input_y: evalY[start:end],
                                                     text_cnn.dropout_keep_prob: 1.0,
                                                     text_cnn.is_training_flag : False})
        eval_loss,eval_acc,eval_counter = eval_loss+curr_eval_loss, eval_acc+curr_eval_acc, eval_counter+1
    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)


def main(_):
    word2index, label2index, trainX, trainY, vaildX, validY, testX, testY = load_data(FLAGS.all_data_h5py, FLAGS.id_index_pkl)

    vocab_size = len(word2index)

    text_cnn = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.label_size, FLAGS.learning_rate,
                            FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len, vocab_size, FLAGS.embed_size)

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

        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        for epoch in range(0, FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])
                    print("trainY[start:end]:",trainY[start:end])
                curr_loss, curr_acc, _ = sess.run([text_cnn.loss_val, text_cnn.accuracy, text_cnn.train_op],
                                                    feed_dict={text_cnn.input_x : trainX[start:end],
                                                               text_cnn.input_y : trainY[start:end],
                                                               text_cnn.dropout_keep_prob : 0.8,
                                                               text_cnn.is_training_flag : True})
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 500 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch, counter, loss/float(counter), acc/float(counter)))

            print("going to increment epoch counter....")
            sess.run(text_cnn.epoch_increment)

            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, text_cnn, vaildX, validY, batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))

                save_path = FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess, save_path, global_step=text_cnn.epoch_step)

        test_loss, test_acc = do_eval(sess, text_cnn, testX, testY, batch_size)
        print("Test Loss:%.3f\tTest Accuracy:%.3f" %(test_loss, test_acc))

        writer.close()


def predict():
    word2index, label2index, trainX, trainY, vaildX, validY, testX, testY = load_data(FLAGS.all_data_h5py, FLAGS.id_index_pkl)
    vocab_size = len(word2index)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        text_cnn = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.label_size, FLAGS.learning_rate,
                            FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len, vocab_size, FLAGS.embed_size)

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
            predictions = sess.run(text_cnn.predictions, feed_dict={text_cnn.input_x: testX[start:end],
                                                                    text_cnn.input_y: testY[start:end],
                                                                    text_cnn.dropout_keep_prob : 1.0,
                                                                    text_cnn.is_training_flag : False})
            if len(predictions) == len(testY[start:end]):
                raw_labels.extend(testY[start:end])
                predicted_labels.extend(predictions)

        print classification_report(raw_labels, predicted_labels)


if __name__ == "__main__":
    if FLAGS.is_training:
        tf.app.run()
    else:
        predict()

