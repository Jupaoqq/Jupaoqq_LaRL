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


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            # if type(hyps[0]) is list:
            #    hyps = [hyp.split() for hyp in hyps[0]]
            # else:
            #    hyps = [hyp.split() for hyp in hyps]

            # refs = [ref.split() for ref in refs]
            hyps = [hyps]
            # Shawn's evaluation
            # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


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
