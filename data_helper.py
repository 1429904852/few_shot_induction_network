"""
data process
"""

import os
import json
import random
import copy
from collections import Counter
from itertools import chain
import io
from typing import Dict, Tuple, Optional, List, Union
import numpy as np


class InductionData(object):
    def __init__(self, output_path, sequence_length, num_classes, num_support,
                 num_queries, num_tasks, num_eval_tasks, embedding_size, stop_word_path, low_freq,
                 word_vector_path, is_training=True):
        """
        init method
        :param output_path: path of train/eval data
        :param num_classes: number of support class
        :param num_support: number of support sample per class
        :param num_queries: number of query sample per class
        :param num_tasks: number of pre-sampling tasks, this will speeding up train
        :param num_eval_tasks: number of pre-sampling tasks in eval stage
        :param stop_word_path: path of stop word file
        :param embedding_size: embedding size
        :param low_freq: frequency of words
        :param word_vector_path: path of word vector file(eg. word2vec, glove)
        :param is_training: bool
        """

        self.__output_path = output_path
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self.__sequence_length = sequence_length
        self.__num_classes = num_classes
        self.__num_support = num_support
        self.__num_queries = num_queries
        self.__num_tasks = num_tasks
        self.__num_eval_tasks = num_eval_tasks
        self.__stop_word_path = stop_word_path
        self.__embedding_size = embedding_size
        self.__low_freq = low_freq
        self.__word_vector_path = word_vector_path
        self.__is_training = is_training

        self.vocab_size = None
        self.word_vectors = None
        self.current_category_index = 0  # record current sample category

        print("stop word path: ", self.__stop_word_path)
        print("word vector path: ", self.__word_vector_path)

    @staticmethod
    def load_data(data_path):
        """
        read train/eval data
        :param data_path:
        :return: dict. {class_name: {sentiment: [[]], }, ...}
        """
        category_files = os.listdir(data_path)
        categories_data = {}
        for category_file in category_files:
            file_path = os.path.join(data_path, category_file)
            sentiment_data = {}
            with io.open(file_path, "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    content, label = line.strip().split("\t")
                    if sentiment_data.get(label, None):
                        sentiment_data[label].append(content.split(" "))
                    else:
                        # {1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]], -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]}
                        sentiment_data[label] = [content.split(" ")]

            # print("task name: ", category_file)
            # print("pos samples length: ", len(sentiment_data["1"]))
            # print("neg samples length: ", len(sentiment_data["-1"]))

            # {
            #   apparel.t2:{
            #               1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]],
            #               -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]
            #   },
            #   apparel.t4:{
            #               1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]],
            #               -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]
            #   },
            # }
            categories_data[category_file] = sentiment_data
        return categories_data

    def remove_stop_word(self, data):
        """
        remove low frequency words and stop words, construct vocab
        :param data: {class_name: {sentiment: [[]], }, ...}
        :return:
        """
        # {
        #   apparel.t2:{
        #               support:{1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]], -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]}
        #               query:{1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]], -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]}
        #   },
        #   apparel.t4:{
        #               support:{1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]], -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]}
        #               query:{1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]], -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]}
        #   },
        # }
        all_words = []
        for category, category_data in data.items():
            for type, data1 in category_data.items():
                for sentiment, sentiment_data in data1.items():
                    all_words.extend(list(chain(*sentiment_data)))
        word_count = Counter(all_words)  # statistic the frequency of words
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # remove low frequency word
        words = [item[0] for item in sort_word_count if item[1] > self.__low_freq]

        return words

    def get_word_vectors(self, vocab):
        """
        load word vector file,
        :param vocab: vocab
        :return:
        """
        pad_vector = np.zeros(self.__embedding_size)  # set the "<pad>" vector to 0
        word_vectors = (1 / np.sqrt(len(vocab) - 1) * (2 * np.random.rand(len(vocab) - 1, self.__embedding_size) - 1))
        word_vectors = np.vstack((pad_vector, word_vectors))

        # load glove vectors
        glove_vector = {}
        with io.open(self.__word_vector_path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line_list = line.strip().split(" ")
                glove_vector[line_list[0]] = line_list[1:]

        for i in range(1, len(vocab)):
            if glove_vector.get(vocab[i], None):
                word_vectors[i, :] = glove_vector[vocab[i]]
            else:
                continue
                # print(vocab[i] + " not exist word vector file")
                # print(" not exist word vector file")

        # # load gensim word2vec vectors
        # if os.path.splitext(self.__word_vector_path)[-1] == ".bin":
        #     word_vec = gensim.models.KeyedVectors.load_word2vec_format(self.__word_vector_path, binary=True)
        # else:
        #     word_vec = gensim.models.KeyedVectors.load_word2vec_format(self.__word_vector_path)
        #
        # for i in range(1, len(vocab)):
        #     try:
        #         vector = word_vec.wv[vocab[i]]
        #         word_vectors[i, :] = vector
        #     except:
        #         print(vocab[i] + "not exist word vector file")

        return word_vectors

    def gen_vocab(self, words):
        """
        generate word_to_index mapping table
        :param words:
        :return:
        """
        if self.__is_training:
            vocab = ["<pad>", "<unk>"] + words

            self.vocab_size = len(vocab)

            if self.__word_vector_path:
                word_vectors = self.get_word_vectors(vocab)
                self.word_vectors = word_vectors
                # save word vector to npy file
                np.save(os.path.join(self.__output_path, "word_vectors.npy"), self.word_vectors)

            word_to_index = dict(zip(vocab, list(range(len(vocab)))))

            # save word_to_index to json file
            with open(os.path.join(self.__output_path, "word_to_index.json"), "w") as f:
                json.dump(word_to_index, f)
        else:
            with open(os.path.join(self.__output_path, "word_to_index.json"), "r") as f:
                word_to_index = json.load(f)

        return word_to_index

    @staticmethod
    def trans_to_index(data, word_to_index):
        """
        transformer token to id
        :param data:
        :param word_to_index:
        :return: {class_name: [[], [], ], ..}
        """
        data_ids = {category: {sentiment: [[word_to_index.get(token, word_to_index["<unk>"]) for token in line]
                                           for line in sentiment_data]
                               for sentiment, sentiment_data in category_data.items()}
                    for category, category_data in data.items()}
        return data_ids

    def choice_support_query(self, task_data):
        """
        randomly selecting support set, query set form a task.
        :param task_data: all data for a task
        :return:
        """
        label_to_index = {"1": 0, "-1": 1}
        if self.__is_training:
            with open(os.path.join(self.__output_path, "label_to_index.json"), "w") as f:
                json.dump(label_to_index, f)

        if self.__is_training:
            pos_samples = task_data["1"]
            neg_samples = task_data["-1"]

            pos_support = random.sample(pos_samples, self.__num_support)
            neg_support = random.sample(neg_samples, self.__num_support)

            pos_others = copy.copy(pos_samples)
            [pos_others.remove(data) for data in pos_support]

            neg_others = copy.copy(neg_samples)
            [neg_others.remove(data) for data in neg_support]

            # print(len(neg_others))
            pos_query = random.sample(pos_others, self.__num_queries)
            neg_query = random.sample(neg_others, self.__num_queries)

            pos_support = self.padding(pos_support)
            neg_support = self.padding(neg_support)
            pos_query = self.padding(pos_query)
            neg_query = self.padding(neg_query)

            support_set = pos_support + neg_support  # [num_classes, num_support, sequence_length]
            query_set = pos_query + neg_query  # [num_classes * num_queries, sequence_length]
            support_set = np.concatenate((support_set, query_set), 0)
            labels = [label_to_index["1"]] * len(pos_query) + [label_to_index["-1"]] * len(neg_query)
            # print(labels)
        else:
            support = task_data["support"]
            query = task_data["query"]

            pos_samples = support["1"]
            # print(len(pos_samples))
            neg_samples = support["-1"]
            # print(len(neg_samples))

            pos_support = random.sample(pos_samples, self.__num_support)
            neg_support = random.sample(neg_samples, self.__num_support)

            pos_samples_query = query["1"]
            # print(len(pos_samples_query))
            neg_samples_query = query["-1"]
            # print(len(neg_samples_query))

            pos_query = random.sample(pos_samples_query, self.__num_queries)
            neg_query = random.sample(neg_samples_query, self.__num_queries)

            pos_support = self.padding(pos_support)
            neg_support = self.padding(neg_support)
            pos_query = self.padding(pos_query)
            neg_query = self.padding(neg_query)

            support_set = pos_support + neg_support  # [num_classes, num_support, sequence_length]
            query_set = pos_query + neg_query  # [num_classes * num_queries, sequence_length]
            support_set = np.concatenate((support_set, query_set), 0)
            labels = [label_to_index["1"]] * len(pos_query) + [label_to_index["-1"]] * len(neg_query)
            # print(labels)
        return support_set, np.asarray(labels)

    def samples(self, data_ids):
        """
        positive and negative sample from raw data
        :param data_ids:
        :return:
        """
        # product name list
        category_list = list(data_ids.keys())
        # print(category_list)

        tasks = []
        if self.__is_training:
            num_tasks = self.__num_tasks
        else:
            num_tasks = self.__num_eval_tasks
        for i in range(num_tasks):
            # randomly choice a category to construct train sample
            support_category = random.choice(category_list)
            support_set, labels = self.choice_support_query(data_ids[support_category])
            tasks.append(dict(support=support_set, labels=labels))
        return tasks

    def remove_stop_word_1(self, data, dev_support, dev_query, test_support, test_query):
        """
        remove low frequency words and stop words, construct vocab
        :param data: {class_name: {sentiment: [[]], }, ...}
        :return:
        """

        all_words = []
        for category, category_data in data.items():
            for sentiment, sentiment_data in category_data.items():
                all_words.extend(list(chain(*sentiment_data)))

        dev_supports = []
        for category, category_data in dev_support.items():
            for sentiment, sentiment_data in category_data.items():
                dev_supports.extend(list(chain(*sentiment_data)))
        all_words.extend(dev_supports)

        dev_querys = []
        for category, category_data in dev_query.items():
            for sentiment, sentiment_data in category_data.items():
                dev_querys.extend(list(chain(*sentiment_data)))
        all_words.extend(dev_querys)

        test_supports = []
        for category, category_data in test_support.items():
            for sentiment, sentiment_data in category_data.items():
                test_supports.extend(list(chain(*sentiment_data)))
        all_words.extend(test_supports)

        test_querys = []
        for category, category_data in test_query.items():
            for sentiment, sentiment_data in category_data.items():
                test_querys.extend(list(chain(*sentiment_data)))
        all_words.extend(test_querys)

        word_count = Counter(all_words)  # statistic the frequency of words
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # remove low frequency word
        words = [item[0] for item in sort_word_count if item[1] > self.__low_freq]

        return words

    def gen_data_1(self, train_file_path, dev_file_path, dev_file_query, test_data_support, test_file_query):
        """
        Generate data that is eventually input to the model
        :return:
        """
        # load data
        train_data = self.load_data(train_file_path)
        dev_support = self.load_data(dev_file_path)
        dev_query = self.load_data(dev_file_query)
        test_support = self.load_data(test_data_support)
        test_query = self.load_data(test_file_query)

        # remove stop word
        words = self.remove_stop_word_1(train_data, dev_support, dev_query, test_support, test_query)
        # print(words)
        word_to_index = self.gen_vocab(words)
        # print(word_to_index)
        data_ids = self.trans_to_index(train_data, word_to_index)
        dev_support_data_ids = self.trans_to_index(dev_support, word_to_index)
        dev_query_data_ids = self.trans_to_index(dev_query, word_to_index)
        test_support_data_ids = self.trans_to_index(test_support, word_to_index)
        test_query_data_ids = self.trans_to_index(test_query, word_to_index)

        np.save(os.path.join("output/induction", "data_ids.npy"), data_ids)
        np.save(os.path.join("output/induction", "dev_support_data_ids.npy"), dev_support_data_ids)
        np.save(os.path.join("output/induction", "dev_query_data_ids.npy"), dev_query_data_ids)
        np.save(os.path.join("output/induction", "test_support_data_ids.npy"), test_support_data_ids)
        np.save(os.path.join("output/induction", "test_query_data_ids.npy"), test_query_data_ids)

        return data_ids

    @staticmethod
    def load_data_1(data_path, file_path_query):
        """
        read train/eval data
        :param data_path:
        :return: dict. {class_name: {sentiment: [[]], }, ...}
        """
        category_files = os.listdir(data_path)
        categories_data = {}
        for category_file in category_files:
            file_path = os.path.join(data_path, category_file)
            sentiment_data = {}
            # sentiment_data_1 = {}
            with io.open(file_path, "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    content, label = line.strip().split("\t")
                    if sentiment_data.get(label, None):
                        sentiment_data[label].append(content.split(" "))
                    else:
                        # {1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]], -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]}
                        sentiment_data[label] = [content.split(" ")]
            # sentiment_data_1["support"] = sentiment_data
            category_file = ".".join(category_file.split(".")[:-1])

            categories_data[category_file] = sentiment_data

        cate_files = os.listdir(file_path_query)
        cate_data = {}
        for cate_file in cate_files:
            file_path = os.path.join(file_path_query, cate_file)
            senti_data = {}
            # senti_data_1 = {}
            with io.open(file_path, "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    content, label = line.strip().split("\t")
                    if senti_data.get(label, None):
                        senti_data[label].append(content.split(" "))
                    else:
                        # {1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]], -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]}
                        senti_data[label] = [content.split(" ")]
            # senti_data_1["query"] = senti_data
            cate_file = ".".join(cate_file.split(".")[:-1])
            cate_data[cate_file] = senti_data

            # {
            #   apparel.t2:{
            #               1: [[ew, ee, ew, ew, we], [ds, f, fd, df, fd]],
            #               -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]
            #   },
            #   apparel.t4:{
            #               1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]],
            #               -1: [[ew,ee,ew,ew,we],[ds,f,fd,df,fd]]
            #   },
            # }

        categories_data_1 = {}
        for key1, value1 in categories_data.items():
            candidate_data = {}
            if key1 in cate_data:
                candidate_data["support"] = value1
                candidate_data["query"] = cate_data[key1]
                categories_data_1[key1] = candidate_data

        return categories_data_1

    @staticmethod
    def trans_to_index_1(data, word_to_index):
        """
        transformer token to id
        :param data:
        :param word_to_index:
        :return: {class_name: [[], [], ], ..}
        """
        data_ids = {category: {type: {sentiment: [[word_to_index.get(token, word_to_index["<unk>"]) for token in line] for line in sentiment_data] for sentiment, sentiment_data in data1.items()} for type, data1 in category_data.items()} for category, category_data in data.items()}
        # print(data_ids)
        return data_ids

    def gen_data(self, file_path, file_path_query):
        """
        Generate data that is eventually input to the model
        :return:
        """
        # load data
        data = self.load_data_1(file_path, file_path_query)
        # print(data)
        # remove stop word
        words = self.remove_stop_word(data)
        # print(words)
        word_to_index = self.gen_vocab(words)
        # print(word_to_index)
        data_ids = self.trans_to_index_1(data, word_to_index)
        # print(data_ids)
        return data_ids

    def padding(self, sentences):
        """
        padding according to the predefined sequence length
        :param sentences:
        :return:
        """
        sentence_pad = [sentence[:self.__sequence_length] if len(sentence) > self.__sequence_length
                        else sentence + [0] * (self.__sequence_length - len(sentence))
                        for sentence in sentences]
        return sentence_pad

    def next_batch(self, data_ids):
        """
        train a task at every turn
        :param data_ids:
        :return:
        """
        tasks = self.samples(data_ids)

        for task in tasks:
            yield task
