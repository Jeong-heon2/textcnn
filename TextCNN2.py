import os
import time
import datetime
from tensorflow import flags
import tensorflow as tf
import numpy as np
import cnn_tool as tool
import pandas as pd
import Mix_Sampling as mix

class TextCNN2(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    <Parameters>
        - sequence_length: 최대 문장 길이
        - num_classes: 클래스 개수
        - vocab_size: 등장 단어 수
        - embedding_size: 각 단어에 해당되는 임베디드 벡터의 차원
        - filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
        - num_filters: 각 filter size 별 filter 수
        - l2_reg_lambda: 각 weights, biases에 대한 l2 regularization 정도
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.F1 = 0

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        """
        <Variable>
            - W: 각 단어의 임베디드 벡터의 성분을 랜덤하게 할당
        """
        # with tf.device('/gpu:0'), tf.name_scope("embedding"):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        try:
            # self.h_pool = tf.concat(3, pooled_outputs)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        except Exception as e:
            print(e)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        # for f1 score
        with tf.name_scope("tp"):
            """
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, "float"))
            
            self.tp = tf.reduce_sum(
                tf.cast(tf.metrics.true_positives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions),
                        "float"), name="tp")
            """
            actuals = tf.argmax(self.input_y, 1)
            ones_like_actuals = tf.ones_like(actuals)
            ones_like_predictions = tf.ones_like(self.predictions)
            self.tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(self.predictions, ones_like_predictions)), "float"))
        with tf.name_scope("tn"):
            """
            self.tn = tf.reduce_sum(
                tf.cast(tf.metrics.true_negatives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions),
                        "float"), name="tn")
            """
            actuals = tf.argmax(self.input_y, 1)

            zeros_like_actuals = tf.zeros_like(actuals)
            zeros_like_predictions = tf.zeros_like(self.predictions)
            self.tn = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(self.predictions, zeros_like_predictions)
                    ),
                    "float"
                )
            )


        with tf.name_scope("fp"):
            """
            self.fp = tf.reduce_sum(
                tf.cast(tf.metrics.false_positives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions),
                        "float"), name="fp")
            """
            actuals = tf.argmax(self.input_y, 1)

            zeros_like_actuals = tf.zeros_like(actuals)
            ones_like_predictions = tf.ones_like(self.predictions)
            self.fp = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(self.predictions, ones_like_predictions)
                    ),
                    "float"
                )
            )

        with tf.name_scope("fn"):
            """
            self.fn = tf.reduce_sum(
                tf.cast(tf.metrics.false_negatives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions),
                        "float"), name="fn")
            """
            actuals = tf.argmax(self.input_y, 1)

            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_predictions = tf.zeros_like(self.predictions)
            self.fn = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(self.predictions, zeros_like_predictions)
                    ),
                    "float"
                )
            )



# data loading
# data_path = 'C:/Users/ratsgo/GoogleDrive/내폴더/textmining/data/watcha_movie_review_spacecorrected.csv'
# data_path = os.getcwd() + "\\data5.csv"
print(datetime.datetime.now().isoformat() + '  데이터로딩 시작')
contents, points = tool.loading_rdata("data5.csv", eng=True, num=True, punc=False)
print(datetime.datetime.now().isoformat() + '  데이터로딩 완료, cut 시작')
contents = tool.cut(contents, cut=2)
print(datetime.datetime.now().isoformat() + '  데이터 cut 완료')

# tranform document to vector
max_document_length = 3000
x, vocabulary, vocab_size = tool.make_input(contents, max_document_length)
print(datetime.datetime.now().isoformat() + '  사전단어수 : %s' % (vocab_size))
y = tool.make_output(points, threshold=1)
print(datetime.datetime.now().isoformat() + '  make_output 완료, train test 나누기 시작')

# divide dataset into train/test set
# 이거 없이 mixsampling 해서 던져줄것임  x_train, x_test, y_train, y_test = tool.divide(x, y, train_prop=0.8)
# todo : mix sampling  df_data를 판다스 포멧으로
x = pd.DataFrame(x)
points = pd.DataFrame(points)
x['label'] = points
x_train, x_test, y_train, y_test = mix.mix_sampling(x, 0.8, 0.2)

print(datetime.datetime.now().isoformat() + '  train, test 나누기 완료')

# Model Hyperparameters
flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of embedded vector (default: 128)")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 128)")
flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#    print("{}={}".format(attr.upper(), value))
# print("")

# 3. train the model and test
# with tf.Graph().as_default():
with tf.device("/cpu:0"):
    cnn = TextCNN2(sequence_length=x_train.shape[1],
                  num_classes=y_train.shape[1],
                  vocab_size=vocab_size,
                  embedding_size=FLAGS.embedding_dim,
                  filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                  num_filters=FLAGS.num_filters,
                  l2_reg_lambda=FLAGS.l2_reg_lambda)

    sess = tf.Session()
    with sess.as_default():

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        '''
        # Summaries for f1 score
        fp_summary = tf.summary.scalar("fp", cnn.fp)
        fn_summary = tf.summary.scalar("fn", cnn.fn)
        recall_summary = tf.summary.scalar("recall", cnn.recall)
        precision_summary = tf.summary.scalar("precision", cnn.precision)
        f1_summary = tf.summary.scalar("f1", cnn.F1)'''

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        try:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

        except Exception as e:
            print(e)


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}"
                  .format(time_str, step, loss, accuracy))

            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0,
            }
            step, summaries, loss, accuracy, tp, tn,  fp, fn = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.tp, cnn.tn, cnn.fp, cnn.fn],
                feed_dict)

            tp = float(tp)
            tn = float(tn)
            fn = float(fn)
            fp = float(fp)
            if writer:
                writer.add_summary(summaries, step)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            return tp, tn, fp, fn, accuracy


        def batch_iter(data, batch_size, shuffle=True):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
            for batch_num in range(num_batches_per_epoch):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]



        # Generate batches
        data = list(zip(x_train, y_train))
        # Training loop. For each batch...
        for epoch in range(FLAGS.num_epochs):
            print(epoch)
            batches = batch_iter(
                data, FLAGS.batch_size, True)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

            #test loop
            print("\nEvaluation: 누적")
            data_test = list(zip(x_test, y_test))
            testpoint = 0
            batches_test = batch_iter(
                data_test, FLAGS.batch_size, False)
            tp_cnt = 0
            tn_cnt = 0
            fp_cnt = 0
            fn_cnt = 0
            for batch in batches_test:
                x_batch, y_batch = zip(*batch)
                tp, tn, fp, fn, acc = dev_step(x_batch, y_batch, writer=dev_summary_writer)
                time_str = datetime.datetime.now().isoformat()
                print("{}:, acc {:g}, tp {:g}, tn {:g}, fp {:g}, fn {:g}"
                      .format(time_str, acc, tp, tn, fp, fn))
                tp_cnt += tp
                tn_cnt += tn
                fp_cnt += fp
                fn_cnt += fn

            recall = tp_cnt / (tp_cnt + fn_cnt)
            precision = tp_cnt / (tp_cnt + fp_cnt)
            f1 = (2 * precision * recall) / (precision + recall)
            acc = (tp_cnt + tn_cnt) / (tp_cnt + fp_cnt + fn_cnt + tn_cnt)
            time_str = datetime.datetime.now().isoformat()
            print("\nEvaluation : 최종")
            print("{}:, acc {:g}, tp {:g}, tn {:g}, fp {:g}, fn {:g}, recall {:g}, precision {:g}, f1 {:g}"
                  .format(time_str, acc, tp_cnt, tn_cnt, fp_cnt, fn_cnt, recall, precision, f1))







