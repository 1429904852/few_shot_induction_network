#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import tensorflow as tf
from model import InductionModel
import numpy as np
import random


def online_predict():
    with open("config.json", "r") as fr:
        config = json.load(fr)
    with open("output/induction/word_to_index.json", "r") as f:
        word2id = json.load(f)
    word_vectors = np.load("output/induction/word_vectors.npy")

    test_support_data_ids = np.load(os.path.join("output/induction", "test_support_data_ids.npy"))

    config["num_classes"] = 2
    model = InductionModel(config=config, vocab_size=len(word2id), word_vectors=word_vectors)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
    max_lens = config["sequence_length"]
    with tf.Session(config=sess_config) as sess:

        # save_path = os.path.join(os.path.abspath(os.getcwd()), config["ckpt_model_path"])
        # checkpoint_prefix = os.path.join(save_path, config["model_name"] + "-500")

        checkpoint_dir = os.path.join(os.path.abspath(os.getcwd()), config["ckpt_model_path"])
        checkpoint_prefix = tf.train.latest_checkpoint(checkpoint_dir)

        model.saver.restore(sess, checkpoint_prefix)

        # pos_sentence = ["i like it", "i love it", "i get it make me so happy", "happy", "so love it"]
        # # neu_sentence = ["hello world", "hello python", "what is it", "are you ok", "hello c++"]
        # neg_sentence = ["it's do bad", "rubbish", "i don't like it", "i don't love it", "shit"]

        test_data_ids = test_support_data_ids.item()
        target_classes = list(test_data_ids.keys())

        # class_name = random.choice(target_classes)
        accuracy = 0

        num_tasks = 100

        for i in range(num_tasks):
            class_name = random.choice(target_classes)
            # for i, class_name in enumerate(target_classes):
            task_data = test_data_ids[class_name]
            pos_samples = task_data["1"]
            print(len(pos_samples))
            neg_samples = task_data["-1"]

            pos_support = random.sample(pos_samples, config["num_support"])
            neg_support = random.sample(neg_samples, config["num_support"])

            pos_support = [sentence[:max_lens] if len(sentence) > max_lens
                       else sentence + [0] * (max_lens - len(sentence))
                       for sentence in pos_support]

            neg_support = [sentence[:max_lens] if len(sentence) > max_lens
                           else sentence + [0] * (max_lens - len(sentence))
                           for sentence in neg_support]

            support_set = pos_support + neg_support

        # pos_ids = []
        # for sentence in pos_sentence:
        #     ids = [word2id[word] if word in word2id else 1 for word in sentence.split(" ")]
        #     if len(ids) < max_lens:
        #         ids = ids + [0] * (max_lens - len(ids))
        #     ids = ids[:max_lens]
        #     pos_ids.append(ids)
        #
        # neg_ids = []
        # for sentence in neg_sentence:
        #     ids = [word2id[word] if word in word2id else 1 for word in sentence.split(" ")]
        #     if len(ids) < max_lens:
        #         ids = ids + [0] * (max_lens - len(ids))
        #     ids = ids[:max_lens]
        #     neg_ids.append(ids)
        #
        # # support = [pos_ids, neu_ids, neg_ids]
        # support = pos_ids + neg_ids

            test_query_data_ids = np.load(os.path.join("output/induction", "test_query_data_ids.npy"))
            class_ = class_name.split(".")
            class_[-1] = "test"
            class_name = ".".join(class_)

            test_query_ids = test_query_data_ids.item()

            task_query_data = test_query_ids[class_name]
            query_pos_samples = task_query_data["1"]
            query_neg_samples = task_query_data["-1"]

            pos_query = random.sample(query_pos_samples, config["num_queries"])
            neg_query = random.sample(query_neg_samples, config["num_queries"])

            pos_query = [sentence[:max_lens] if len(sentence) > max_lens
                           else sentence + [0] * (max_lens - len(sentence))
                           for sentence in pos_query]

            neg_query = [sentence[:max_lens] if len(sentence) > max_lens
                           else sentence + [0] * (max_lens - len(sentence))
                           for sentence in neg_query]

            query_set = pos_query + neg_query

        # # neu_sentence = ["hello world", "hello python", "what is it", "are you ok", "hello c++"]
        # # neu_sentence_1 = ["fsdf world", "hello fsafds", "fsafd is it", "fsadf you ok", "fsdf c++"]
        # #
        # # neu_ids = []
        # # for sentence in neu_sentence:
        # #     ids = [word2id[word] if word in word2id else 1 for word in sentence.split(" ")]
        # #     if len(ids) < max_lens:
        # #         ids = ids + [0] * (max_lens - len(ids))
        # #     ids = ids[:max_lens]
        # #     neu_ids.append(ids)
        # #
        # # neu_ids_1 = []
        # # for sentence in neu_sentence_1:
        # #     ids = [word2id[word] if word in word2id else 1 for word in sentence.split(" ")]
        # #     if len(ids) < max_lens:
        # #         ids = ids + [0] * (max_lens - len(ids))
        # #     ids = ids[:max_lens]
        # #     neu_ids_1.append(ids)
        #
        # query = neu_ids + neu_ids_1

            support_set = np.concatenate((support_set, query_set), 0)
            label_to_index = {"1": 0, "-1": 1}
            labels = [label_to_index["1"]] * len(pos_query) + [label_to_index["-1"]] * len(neg_query)
            labels = np.asarray(labels)

            # support_set = support + queries
            batch1 = {"support": support_set, "labels": labels}

            # batch1 = {"queries": queries, "support": support}
            predict, scores, acc = model.infer(sess, batch1)

            accuracy += acc
            print("===============================")
            # print(predict, scores)

            print("test:  class_name: {}, acc: {:.4f}".format(class_name, acc))
        print("total:  acc: {:.4f}".format(accuracy / num_tasks))
        # while True:
        #     queries = []
        #     for i in range(1):
        #         # neg_sentence = ["it's do bad", "rubbish", "i don't like it", "i don't love it", "shit"]
        #         sentence = input("input sentence:")
        #         ids = [word2id[word] if word in word2id else 1 for word in sentence.split(" ")]
        #         if len(ids) < max_lens:
        #             ids = ids + [0] * (max_lens - len(ids))
        #         ids = ids[:max_lens]
        #         queries.append(ids)
        #
        #     support_set = np.concatenate((support, queries), 0)
        #
        #     # support_set = support + queries
        #     batch1 = {"support": support_set}
        #
        #     # batch1 = {"queries": queries, "support": support}
        #     predict, scores = model.infer(sess, batch1)
        #
        #     print("===============================")
        #     print(predict, scores)

if __name__ == "__main__":
    online_predict()
