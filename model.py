import tensorflow as tf


class InductionModel(object):
    def __init__(self, config, vocab_size, word_vectors):
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        self.num_classes = self.config["num_classes"]

        # [num_classes, num_support, sequence_length]
        self.support = tf.placeholder(tf.int32, [None, None], name="support")
        # [num_classes * num_queries, sequence_length]
        # self.queries = tf.placeholder(tf.int32, [None, None], name="queries")
        # [num_classes * num_queries]
        self.labels = tf.placeholder(tf.int32, [None], name="labels")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout

        self.global_step = tf.Variable(name="global_step", initial_value=0, trainable=False)

        self.l2_loss = tf.constant(0.0)  # l2 regulation

        # construct graph
        self.model_structure()
        # Initialize the object that saves the model
        self.init_saver()

    def model_structure(self):
        # embedding layer
        with tf.name_scope("encoder"):
            # Initialization of Word Embedding Matrix Using Pre-trained Word Vectors
            if self.word_vectors is not None:
                embedding_w = tf.constant(self.word_vectors, dtype=tf.float32, name='word_embedding')
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config["embedding_size"]],
                                              initializer=tf.keras.initializers.glorot_normal())

            # support embedding. dimension: [2*num_classes*num_support, sequence_length, embedding_size]
            support_embedded = tf.nn.embedding_lookup(embedding_w, self.support, name="support_embedded")
            support_embedded = tf.nn.dropout(support_embedded, keep_prob=self.keep_prob)

            support_embedded_reshape = tf.reshape(support_embedded, [-1, self.config["sequence_length"], self.config["embedding_size"]])

            cell_fw = tf.contrib.rnn.LSTMCell
            cell_bw = tf.contrib.rnn.LSTMCell
            support_output, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw(self.config["hidden_sizes"]),
                cell_bw=cell_bw(self.config["hidden_sizes"]),
                inputs=support_embedded_reshape,
                dtype=tf.float32,
                scope="lstm_1"
            )

            # concat frontward and backward output of lstm
            # [2 * num_classes * num_support, sequence_length, hidden_size * 2]
            support_embedded_reshape = tf.concat(support_output, -1)
            # support_embedded_reshape = tf.nn.dropout(support_embedded_reshape, keep_prob=self.keep_prob)
            # [num_classes * num_queries, sequence_length, hidden_size * 2]
            # queries_embedded = tf.concat(queries_output, -1)
            # [2 * num_classes * num_support, hidden_size * 2]
            support_final_output = self._attention(support_embedded_reshape, scope_name="support")
            # [num_classes * num_queries, hidden_size * 2]
            # queries_final_output = self._attention(queries_embedded, scope_name="queries")
            # support_final_output = tf.nn.dropout(support_final_output, keep_prob=self.keep_prob)
            support_encoder = tf.slice(support_final_output, [0, 0],
                                       [self.num_classes * self.config["num_support"], self.config["hidden_sizes"] * 2])
            query_encoder = tf.slice(support_final_output, [self.num_classes * self.config["num_support"], 0],
                                     [self.num_classes * self.config["num_queries"], self.config["hidden_sizes"] * 2])

        # induction module be used to generate class vectors
        with tf.name_scope("induction_module"):
            support_class = self.dynamic_routing(tf.reshape(support_encoder, [self.num_classes, self.config["num_support"], -1]))

        with tf.name_scope("relation_module"):
            self.scores = self.neural_tensor_layer(support_class, query_encoder)
            # self.predictions = tf.argmax(self.scores, axis=-1, name="predictions")
            self.predictions = tf.argmax(name="predictions", input=self.scores, axis=1)

        with tf.name_scope("loss"):
            labels_one_hot = tf.one_hot(self.labels, self.num_classes, dtype=tf.float32)
            losses = tf.losses.mean_squared_error(labels=labels_one_hot, predictions=self.scores)
            # self.loss = losses

            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v)
                 for v in tf.trainable_variables() if 'bias' not in v.name]) * self.config["l2_reg_lambda"]
            self.loss = losses + l2_losses

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
            self.accuracy = tf.reduce_mean(name="accuracy", input_tensor=tf.cast(correct_prediction, tf.float32))

        with tf.name_scope("train_op"):
            # define optimizer
            # self.train_op = self.get_optimizer().minimize(self.loss)
            optimizer = self.get_optimizer()
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            # gradient clip
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    def dynamic_routing(self, support_encoding, iter_routing=3):
        """
        the dynamic routing algorithm
        :param support_encoding:
        :param iter_routing: number of iterate
        :return:
        """

        num_classes = self.num_classes
        num_support = self.config["num_support"]
        encode_size = self.config["hidden_sizes"] * 2

        # init dynamic routing values, weights of samples per class. [num_classes, num_support]
        init_b = tf.constant(0.0, dtype=tf.float32, shape=[num_classes, num_support])

        # transformer matrix, mapping input to another space. [encode_size, encode_size]
        w_s = tf.get_variable("w_s", shape=[encode_size, encode_size], dtype=tf.float32,
                              initializer=tf.keras.initializers.glorot_normal())

        # Iterating to update dynamic routing values
        for r_iter in range(iter_routing):
            # normalization dynamic routing values. [num_classes, num_support, 1]
            norm_b = tf.nn.softmax(tf.reshape(init_b, [num_classes, num_support, 1]), axis=1)
            # mapping support to another space. [num_classes, num_support, encoder_size]
            support_trans = tf.reshape(tf.matmul(tf.reshape(support_encoding, [-1, encode_size]), w_s),
                                       [num_classes, num_support, encode_size])

            # weighted sum. [num_classes, encoder_size]
            c_i = tf.reduce_sum(tf.multiply(support_trans, norm_b), axis=1)
            # squash activate function
            c_squared_norm = tf.reduce_sum(tf.square(c_i), 1, keepdims=True)
            scalar_factor = c_squared_norm / (1 + c_squared_norm) / tf.sqrt(c_squared_norm + 1e-9)
            c_squashed = scalar_factor * c_i  # element-wise

            # Calculate the dot between samples and class vectors for per class
            c_e_dot = tf.matmul(support_trans, tf.reshape(c_squashed, [num_classes, encode_size, 1]))

            # update dynamic routing values
            init_b += tf.reshape(c_e_dot, [num_classes, num_support])

        return c_squashed

    def neural_tensor_layer(self, class_vector, query_encoder):
        """
        calculate relation scores
        :param class_vector: class vectors
        :param query_encoder: query set encoding matrix. [num_classes * num_queries, encode_size]
        :return:
        """

        """neural tensor layer (NTN)"""
        # [num_classes, hidden_size*2] [num_classes * query_num_per_class, hidden_size * 2]
        C, H = class_vector.shape
        # print("class_vector shape:", class_vector.shape)
        # print("query_encoder shape:", query_encoder.shape)
        M = tf.get_variable("M", [H, H, self.config["layer_size"]], dtype=tf.float32,
                            initializer=tf.keras.initializers.glorot_normal())
        mid_pro = []
        for slice in range(self.config["layer_size"]):
            # [num_classes, hidden_size*2] [num_classes * query_num_per_class, hidden_size * 2]
            # [num_classes, query_num_per_class]
            slice_inter = tf.matmul(tf.matmul(class_vector, M[:, :, slice]), query_encoder, transpose_b=True)  # (C,Q)
            # [query_num_per_class, num_classes, out_size]
            mid_pro.append(slice_inter)
        # [query_num_per_class, num_classes, out_size]
        tensor_bi_product = tf.concat(mid_pro, axis=0)  # (C*K,Q)
        # print("tensor_bi_product shape:{}".format(tensor_bi_product.shape))

        # [out_size, num_classes * query_num_per_class]
        V = tf.nn.relu(tf.transpose(tensor_bi_product))
        W = tf.get_variable("w", [C * self.config["layer_size"], C], dtype=tf.float32,
                            initializer=tf.keras.initializers.glorot_normal())
        b = tf.get_variable("b", [C], dtype=tf.float32,
                            initializer=tf.keras.initializers.glorot_normal())
        # [out_size, num_classes * query_num_per_class]
        probs = tf.nn.sigmoid(tf.matmul(V, W) + b)  # (Q,C)
        return probs

        # num_classes = self.num_classes
        # encode_size = self.config["hidden_sizes"][-1] * 2
        # layer_size = self.config["layer_size"]
        #
        # M = tf.get_variable("M", [encode_size, encode_size, layer_size], dtype=tf.float32,
        #                     initializer=tf.truncated_normal_initializer(stddev=(2 / encode_size) ** 0.5))
        #
        # # [[class1, class2, ..], [class1, class2, ..], ... layer_size]
        # all_mid = []
        # for i in range(layer_size):
        #     # [num_classes, num_classes * num_queries]
        #     slice_mid = tf.matmul(tf.matmul(class_vector, M[:, :, i]), query_encoder, transpose_b=True)
        #     all_mid.append(tf.split(slice_mid, [1] * num_classes, axis=0))
        #
        # # [[1, 2, .. layer_size], ... class_n]
        # all_mid = [[mid[j] for mid in all_mid] for j in range(len(all_mid[0]))]
        #
        # # [layer_size, num_classes * num_queries]
        # all_mid_concat = [tf.concat(mid, axis=0) for mid in all_mid]
        #
        # # [num_classes * num_queries, layer_size]
        # all_mid_transpose = [tf.nn.relu(tf.transpose(mid)) for mid in all_mid_concat]
        #
        # relation_w = tf.get_variable("relation_w", [layer_size, 1], dtype=tf.float32,
        #                              initializer=tf.glorot_normal_initializer())
        # relation_b = tf.get_variable("relation_b", [1], dtype=tf.float32,
        #                              initializer=tf.glorot_normal_initializer())
        #
        # scores = []
        # for mid in all_mid_transpose:
        #     score = tf.nn.sigmoid(tf.matmul(mid, relation_w) + relation_b)
        #     scores.append(score)
        #
        # # [num_classes * num_queries, num_classes]
        # scores = tf.concat(scores, axis=-1)
        #
        # return scores

    def _attention(self, H, scope_name):
        """
        attention for the final output of Lstm
        :param H: [batch_size, sequence_length, hidden_size * 2]
        """
        # with tf.variable_scope(scope_name):
        #     _, sequence_length, hidden_size = H.shape
        #     x_proj = tf.layers.Dense(hidden_size)(H)
        #     x_proj = tf.nn.tanh(x_proj)
        #     u_w = tf.get_variable('W_a2', shape=[hidden_size, 1],
        #                           dtype=tf.float32, initializer=tf.keras.initializers.glorot_normal())
        #     x = tf.tensordot(x_proj, u_w, axes=1)
        #     alphas = tf.nn.softmax(x, axis=1)
        #     print("alphas shape", alphas.shape)
        #     output = tf.matmul(tf.transpose(H, [0, 2, 1]), alphas)
        #     output = tf.squeeze(output, -1)
        #     return output

        with tf.variable_scope(scope_name):
            hidden_size = self.config["hidden_sizes"] * 2
            attention_size = self.config["attention_size"]
            w_1 = tf.get_variable("w_1", shape=[hidden_size, attention_size], initializer=tf.keras.initializers.glorot_normal())
            w_2 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

            # Nonlinear conversion for LSTM output, [batch_size * sequence_length, attention_size]
            M = tf.tanh(tf.matmul(tf.reshape(H, [-1, hidden_size]), w_1))

            # calculate weights, [batch_size, sequence_length]
            weights = tf.reshape(tf.matmul(M, tf.reshape(w_2, [-1, 1])), [-1, self.config["sequence_length"]])

            # softmax normalization, [batch_size, sequence_length]
            alpha = tf.nn.softmax(weights, axis=-1)

            # calculate weighted sum
            output = tf.reduce_sum(H * tf.reshape(alpha, [-1, self.config["sequence_length"], 1]), axis=1)

            return output

    def get_optimizer(self):
        """
        define optimizer
        :return:
        """
        optimizer = None
        if self.config["optimization"] == "adam":
            optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "Adagrad":
            optimizer = tf.train.AdagradOptimizer(self.config["learning_rate"])
        return optimizer

    def init_saver(self):
        """
        init model saver object
        :return:
        """
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, dropout_prob):
        """
        train model method
        :param sess: session object of tensorflow
        :param batch: train batch data
        :param dropout_prob: dropout keep prob
        :return: loss, predict result
        """
        # print(batch["support"])
        # print(batch["labels"])
        feed_dict = {self.support: batch["support"],
                     self.labels: batch["labels"],
                     self.keep_prob: dropout_prob}

        _, loss, predictions, acc = sess.run([self.train_op, self.loss, self.predictions, self.accuracy],
                                        feed_dict=feed_dict)
        return loss, predictions, acc

    def eval(self, sess, batch):
        """
        evaluate model method
        :param sess: session object of tensorflow
        :param batch: eval batch data
        :return: loss, predict result
        """
        feed_dict = {self.support: batch["support"],
                     self.labels: batch["labels"],
                     self.keep_prob: 1.0}

        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def infer(self, sess, batch):
        """
        infer model method
        :param sess:
        :param batch:
        :return: predict result
        """
        feed_dict = {self.support: batch["support"],
                     self.labels: batch["labels"],
                     self.keep_prob: 1.0}

        predict, scores, acc = sess.run([self.predictions, self.scores, self.accuracy], feed_dict=feed_dict)

        return predict, scores, acc
