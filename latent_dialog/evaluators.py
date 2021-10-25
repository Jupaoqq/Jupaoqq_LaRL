from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math
import latent_dialog.normalizer.delexicalize as delex
from latent_dialog.utils import get_tokenize
from collections import Counter
from nltk.util import ngrams
from latent_dialog.corpora import SYS, USR, BOS, EOS
import json
from latent_dialog.normalizer.delexicalize import normalize
import os
import random
import logging


class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BleuEvaluator(BaseEvaluator):
    def __init__(self, data_name):
        self.data_name = data_name
        self.labels = list()
        self.hyps = list()

    def initialize(self):
        self.labels = list()
        self.hyps = list()

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def get_report(self):
        tokenize = get_tokenize()
        print('Generate report for {} samples'.format(len(self.hyps)))
        refs, hyps = [], []
        for label, hyp in zip(self.labels, self.hyps):
            # label = label.replace(EOS, '')
            # hyp = hyp.replace(EOS, '')
            # ref_tokens = tokenize(label)[1:]
            # hyp_tokens = tokenize(hyp)[1:]
            ref_tokens = tokenize(label)
            hyp_tokens = tokenize(hyp)
            refs.append([ref_tokens])
            hyps.append(hyp_tokens)
        bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
        report = '\n===== BLEU = %f =====\n' % (bleu,)
        return '\n===== REPORT FOR DATASET {} ====={}'.format(self.data_name, report)

    def distinct_metrics(self, outs):
        # outputs is a list which contains several sentences, each sentence contains several words
        unigram_count = 0
        bigram_count = 0
        trigram_count=0
        quagram_count=0
        unigram_set = set()
        bigram_set = set()
        trigram_set=set()
        quagram_set=set()
        for sen in outs:
            for word in sen:
                unigram_count += 1
                unigram_set.add(word)
            for start in range(len(sen) - 1):
                bg = str(sen[start]) + ' ' + str(sen[start + 1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(len(sen)-2):
                trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                trigram_count+=1
                trigram_set.add(trg)
            for start in range(len(sen)-3):
                quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                quagram_count+=1
                quagram_set.add(quag)
        dis1 = len(unigram_set) / len(outs) #unigram_count
        dis2 = len(bigram_set) / len(outs) #bigram_count
        dis3 = len(trigram_set)/len(outs) #trigram_count
        dis4 = len(quagram_set)/len(outs) #quagram_count
        return dis1, dis2, dis3, dis4

    def rollout_recall(self, pred, actual):
        # print("pred and actual")
        # print(pred)
        # print(actual)
        total_liked = 0
        l = []
        for key, value in actual.items():
            if value['liked'] == '1':
                total_liked = total_liked + 1

        for i in pred:
            if actual.get(i)['liked'] == '1':
                l.append(1)
            else:
                l.append(0)

        r1 = -1
        r5 = -1
        r10 = -1
        if total_liked > 0:
            if len(pred) > 0:
                r1 = l[:1].count(1) / total_liked
            if len(pred) > 4:
                r5 = l[:5].count(1) / total_liked
            if len(pred) > 9:
                r10 = l[:10].count(1) / total_liked
        # print("r1, r5, r10")
        # print(r1)
        # print(r5)
        # print(r10)

        return r1, r5, r10




