#encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import pickle
import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import bert_model as modeling

reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("all_data_h5py","../../data/short_text_classification/data_to_id.h5","all training data")
tf.app.flags.DEFINE_string("id_index_pkl","../../data/short_text_classification/vocabulary_dict.pkl","word to id, and type to id")
tf.app.flags.DEFINE_string("ckpt_dir","../../data/short_text_classification/bert_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_string("summary_dir","../../data/short_text_classification/summary_dir/","summary log path")
tf.app.flags.DEFINE_float("learning_rate", 0.0001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_boolean("is_training", True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 2,"epoch numbers")

# below hyper-parameter is for bert model
# to train a big model,                     use hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072
# to train a middel size model, train fast. use hidden_size=128, num_hidden_layers=4, num_attention_heads=8, intermediate_size=1024
tf.app.flags.DEFINE_integer("hidden_size", 128,"hidden size") # 768
tf.app.flags.DEFINE_integer("num_hidden_layers", 2,"number of hidden layers") # 12--->4
tf.app.flags.DEFINE_integer("num_attention_heads", 4,"number of attention headers") # 12
tf.app.flags.DEFINE_integer("intermediate_size", 256,"intermediate size of hidden layer") # 3072-->512
tf.app.flags.DEFINE_integer("max_seq_length", 20,"max sequence length")


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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels,
                num_labels, use_one_hot_embeddings,reuse_flag=False):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    
    output_layer = model.get_pooled_output()
    print("output_layer:",output_layer.shape)
    hidden_size = output_layer.shape[-1].value
    with tf.variable_scope("weights",reuse=reuse_flag):
        output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())
    
    with tf.variable_scope("loss"):
        def apply_dropout_last_layer(output_layer):
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            return output_layer

        def not_apply_dropout(output_layer):
            return output_layer

        output_layer = tf.cond(is_training, lambda: apply_dropout_last_layer(output_layer), lambda:not_apply_dropout(output_layer))
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        print("output_layer:",output_layer.shape,";output_weights:",output_weights.shape,";logits:",logits.shape)
        probabilities = tf.nn.softmax(logits, dim=-1)
        print("logits.shape: ", logits.shape)
        loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        print("loss_batch.shape: ", loss_batch.shape)
        loss = tf.reduce_mean(loss_batch)

        predictions = tf.argmax(logits, 1, name="predictions")
        correct_prediction = tf.equal(tf.cast(predictions, tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    return loss, logits, probabilities, predictions, accuracy, model


def get_input_mask_segment_ids(train_x_batch, cls_id):
    batch_size, max_sequence_length = train_x_batch.shape
    input_mask = np.ones((batch_size, max_sequence_length), dtype=np.int32)
    # set 0 for token in padding postion
    for i in range(batch_size):
        input_x_ = train_x_batch[i]
        input_x = list(input_x_)
        for j in range(len(input_x)):
            if input_x[j] == 0:
                input_mask[i][j:] = 0
                break
    # insert CLS token for classification
    input_ids = np.zeros((batch_size, max_sequence_length), dtype=np.int32)
    for k in range(batch_size):
        input_id_list = list(train_x_batch[k])
        input_id_list.insert(0, cls_id)
        del input_id_list[-1]     # 舍弃最后一位是否合理？？？
        input_ids[k] = input_id_list

    segment_ids = np.ones((batch_size, max_sequence_length), dtype=np.int32)
    return input_mask, segment_ids, input_ids


def do_eval(sess, input_ids, input_mask, segment_ids, label_ids, is_training,
        loss, probabilities, accuracy, validX, validY, num_labels, batch_size, cls_id):
    number_examples = len(validX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        input_mask_, segment_ids_, input_ids_ = get_input_mask_segment_ids(validX[start:end], cls_id)
        feed_dict = {input_ids: input_ids_, input_mask:input_mask_, segment_ids:segment_ids_,
                     label_ids: validY[start:end], is_training:False}
        curr_eval_loss, curr_eval_acc = sess.run([loss, accuracy], feed_dict)
        eval_loss, eval_acc, eval_counter = eval_loss+curr_eval_loss, eval_acc+curr_eval_acc, eval_counter+1
    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)


def main(_):
    # 加载数据
    word2index, label2index, trainX, trainY, validX, validY, testX, testY = load_data(FLAGS.all_data_h5py, FLAGS.id_index_pkl)

    vocab_size = len(word2index)
    print("bert model.vocab_size:", vocab_size)
    num_labels = len(label2index)
    print("num_labels:", num_labels)
    cls_id = word2index['CLS']
    print("id of 'CLS':", word2index['CLS'])
    num_examples, FLAGS.max_seq_length = trainX.shape
    print("num_examples of training:", num_examples, "; max_seq_length:", FLAGS.max_seq_length)

    # 构建计算图
    bert_config = modeling.BertConfig(vocab_size=vocab_size, hidden_size=FLAGS.hidden_size, num_hidden_layers=FLAGS.num_hidden_layers,
                                      num_attention_heads=FLAGS.num_attention_heads, intermediate_size=FLAGS.intermediate_size)

    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_ids")
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_mask")
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length],name="segment_ids")
    label_ids = tf.placeholder(tf.int32, [None], name="label_ids")
    is_training = tf.placeholder(tf.bool, name="is_training")

    use_one_hot_embeddings = False
    loss, logits, probabilities, predictions, accuracy, model = create_model(bert_config, is_training, input_ids, input_mask,
                                                            segment_ids, label_ids, num_labels, use_one_hot_embeddings)

    global_step = tf.Variable(0, trainable=False, name="Global_Step")
    train_op = tf.contrib.layers.optimize_loss(loss, global_step=global_step, learning_rate=FLAGS.learning_rate,
                        optimizer="Adam", clip_gradients=3.0)
    
    # 训练
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
        print("Checkpoint Exists. Restoring Variables from Checkpoint.")
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

    writer = tf.summary.FileWriter(FLAGS.summary_dir, tf.get_default_graph())

    number_of_training_data = len(trainX)
    iteration = 0
    curr_epoch = 0
    batch_size = FLAGS.batch_size
    for epoch in range(curr_epoch, FLAGS.num_epochs):
        loss_total, acc, counter = 0.0, 0, 0
        for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
            iteration = iteration + 1
            input_mask_, segment_ids_, input_ids_ = get_input_mask_segment_ids(trainX[start:end], cls_id)
            feed_dict = {input_ids: input_ids_, input_mask: input_mask_, segment_ids:segment_ids_,
                         label_ids: trainY[start:end], is_training:True}
            curr_loss, curr_acc, _ = sess.run([loss, accuracy, train_op], feed_dict)
            loss_total, acc, counter = loss_total + curr_loss, acc + curr_acc, counter + 1
            if counter % 30 == 0:
                print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch, counter, loss_total/float(counter), acc/float(counter)))
            
            if start!=0 and start % (1000 * FLAGS.batch_size) == 0:
                eval_loss, eval_acc = do_eval(sess, input_ids, input_mask, segment_ids, label_ids, is_training, loss,
                                              probabilities, accuracy, validX, validY, num_labels, batch_size, cls_id)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))

                save_path = FLAGS.ckpt_dir + "model.ckpt"
                print("Going to save model..")
                saver.save(sess, save_path, global_step=epoch)
    
    test_loss, test_acc = do_eval(sess, input_ids, input_mask, segment_ids, label_ids, is_training, loss,
                                  probabilities, accuracy, testX, testY, num_labels, batch_size, cls_id)
    print("Test Loss:%.3f\tTest Accuracy: %.3f" % (test_loss, test_acc ))

    writer.close()


def predict():
    # 加载数据
    word2index, label2index, trainX, trainY, validX, validY, testX, testY = load_data(FLAGS.all_data_h5py, FLAGS.id_index_pkl)

    vocab_size = len(word2index)
    num_labels = len(label2index)
    cls_id = word2index['CLS']
    num_examples, FLAGS.max_seq_length = trainX.shape

    # 构建计算图
    bert_config = modeling.BertConfig(vocab_size=vocab_size, hidden_size=FLAGS.hidden_size, num_hidden_layers=FLAGS.num_hidden_layers,
                                      num_attention_heads=FLAGS.num_attention_heads, intermediate_size=FLAGS.intermediate_size)

    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_ids")
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_mask")
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length],name="segment_ids")
    label_ids = tf.placeholder(tf.int32, [None], name="label_ids")
    is_training = tf.placeholder(tf.bool, name="is_training")

    use_one_hot_embeddings = False
    loss, logits, probabilities, predictions, accuracy, model = create_model(bert_config, is_training, input_ids, input_mask,
                                                            segment_ids, label_ids, num_labels, use_one_hot_embeddings)

    global_step = tf.Variable(0, trainable=False, name="Global_Step")
    train_op = tf.contrib.layers.optimize_loss(loss, global_step=global_step, learning_rate=FLAGS.learning_rate,
                        optimizer="Adam", clip_gradients=3.0)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config = gpu_config) as sess:
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Checkpoint Exists. Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

        print("test_X.shape:", testX.shape)
        print("test_Y.shape:", testY.shape)

        raw_labels = []
        predicted_labels = []
        number_examples = len(testX)
        batch_size = FLAGS.batch_size
        for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
            input_mask_, segment_ids_, input_ids_ = get_input_mask_segment_ids(testX[start:end], cls_id)
            feed_dict = {input_ids: input_ids_, input_mask: input_mask_, segment_ids:segment_ids_,
                         label_ids: trainY[start:end], is_training:False}
            pred = sess.run(predictions, feed_dict=feed_dict)
            if len(pred) == len(testY[start:end]):
                raw_labels.extend(testY[start:end])
                predicted_labels.extend(pred)

        print(classification_report(raw_labels, predicted_labels))

if __name__ == "__main__":
    if FLAGS.is_training:
        tf.app.run()
    else:
        predict()
