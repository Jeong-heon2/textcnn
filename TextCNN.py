import os
import time
import datetime
from tensorflow import flags
import tensorflow as tf
import numpy as np
import cnn_tool as tool


class TextCNN(object):
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
        #with tf.device('/gpu:0'), tf.name_scope("embedding"):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #with tf.device('/cpu:0'), tf.name_scope("embedding"):
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
            #self.h_pool = tf.concat(3, pooled_outputs)
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

        with tf.name_scope("num_correct"):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, "float"))
        
        with tf.name_scope("fp"):
            self.fp = tf.reduce_sum(tf.cast(tf.metrics.false_positives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions), "float"), name="fp")

        with tf.name_scope("fn"):
            self.fn = tf.reduce_sum(tf.cast(tf.metrics.false_negatives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions), "float"), name="fn")
        
        with tf.name_scope("recall"):

            self.recall = self.num_correct / (self.num_correct + self.fn)

        with tf.name_scope("precision"):

            self.precision = self.num_correct / (self.num_correct + self.fp)

        with tf.name_scope("F1"):
            """
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            num_correct = tf.reduce_sum(tf.cast(correct, "float"))

            fn = tf.reduce_sum(
                tf.cast(tf.metrics.false_negatives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions),
                        "float"), name="fn")

            fp = tf.reduce_sum(
                tf.cast(tf.metrics.false_positives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions),
                        "float"), name="fp")
            recall = num_correct / (num_correct + fn)
            precision = num_correct / (num_correct + fp)
            """

            self.F1 = (2 * self.precision * self.recall) / (self.precision + self.recall)



# data loading
#data_path = 'C:/Users/ratsgo/GoogleDrive/내폴더/textmining/data/watcha_movie_review_spacecorrected.csv'
#data_path = os.getcwd() + "\\data5.csv"
print(datetime.datetime.now().isoformat() + '  데이터로딩 시작')
contents, points = tool.loading_rdata("data5.csv", eng=True, num=True, punc=False)
print(datetime.datetime.now().isoformat() + '  데이터로딩 완료, cut 시작')
contents = tool.cut(contents,cut=2)
print(datetime.datetime.now().isoformat() + '  데이터 cut 완료')
# tranform document to vector
max_document_length = 3000
x, vocabulary, vocab_size = tool.make_input(contents, max_document_length)
print(datetime.datetime.now().isoformat() + '  사전단어수 : %s' % (vocab_size))
y = tool.make_output(points,threshold=1)
print(datetime.datetime.now().isoformat() + '  make_output 완료, train test 나누기 시작')
# divide dataset into train/test set
# 이거 없이 mix머시기 해서 던져줄것임
x_train, x_test, y_train, y_test = tool.divide(x, y, train_prop=0.8)
print(datetime.datetime.now().isoformat() + '  train, test 나누기 완료')
# Model Hyperparameters
flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of embedded vector (default: 128)")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
#    print("{}={}".format(attr.upper(), value))
#print("")

# 3. train the model and test
#with tf.Graph().as_default():
with tf.device("/cpu:0"):
    cnn = TextCNN(sequence_length=x_train.shape[1],
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
            init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
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
            step, summaries, loss, accuracy, tp, fp, fn, recall, precision, f1 = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.fp, cnn.fn, cnn.recall, cnn.precision, cnn.F1],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, tp {:g}, fp {:g}, fn {:g}, recall {:g}, precision {:g}, f1 {:g}"
                  .format(time_str, step, loss, accuracy, tp, fp, fn, recall, precision, f1))

            tf.metrics.false_negatives.__init__()
            tf.metrics.false_positives.__init__()
            if writer:
                writer.add_summary(summaries, step)


        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]


        # Generate batches
        print(datetime.datetime.now().isoformat() + '  데이터 bach , shuffle 시작')
        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        print(datetime.datetime.now().isoformat() + '  데이터 bach , shuffle 완료')
        testpoint = 0
        # Training loop. For each batch...

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                if testpoint + 100 < len(x_test):
                    testpoint += 100
                else:
                    testpoint = 0
                print("\nEvaluation:")
                dev_step(x_test[testpoint:testpoint+100], y_test[testpoint:testpoint+100], writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


