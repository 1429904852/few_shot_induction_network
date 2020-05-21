# few-shot-learning

复现论文《Few-Shot Text Classiﬁcation with Induction Network》

## Dataset: Amazon Review Sentiment Classification

| Train Tasks | Dev Tasks|Test Tasks|
| ------| ------|------|
| 19 * 3 = 57 | 4 * 3 = 12 |4 * 3 = 12 |

* Data for train, dev and test:
    - Train data: `*.train` files in train domains.
    
    - Dev data: `*.trian` files in test domains as support data and `*.dev` files in test domains as query data.
    
    - Test data: `*.trian` files in test domains as support data and `*.test` files in test domains as query data.
* embedding:
  * please download glove.6B.300d.txt and put it in /word_embedded

## Model

* Encoder Module: bi-direction recurrent neural network with self-attention.
* Induction Module: dynamic routing induction algorithm.
* Relation Module: measure the correlation between each pair of query and class and output the relation scores.

## Train, Dev and Test

```angularjs
python trainer.py
```

* Training Strategy: episode-based meta training

* Dev while training and record dev accuracy.

```angularjs
python predict.py
```
* Pick the checkpoint with the highest dev accuracy as the best model and test on it.

## Reference

This is an implementation of the paper [Few-Shot Text Classification with Induction Network](https://arxiv.org/abs/1902.10482).
