#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluating.py
@Time    :   2021/02/24 11:35:55
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

from collections import Counter
from src.utils import flatten_list


class Metrics(object):
    def __init__(self, gloden_tags, pred_tags, remove_o=False):
        self.golden_tags = flatten_list(gloden_tags)
        self.pred_tags = flatten_list(pred_tags)

        if remove_o:
            self.remove_otags()

        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.pred_tags_count = Counter(self.pred_tags)
        self.gold_tags_count = Counter(self.golden_tags)
        self.precision_scores = self.cal_precision()
        self.recall_scores = self.cal_recall()
        self.f1_scores = self.cal_f1()

    def cal_precision(self):
        precision_scores = {}
        for tag in self.tagset:
            precision_scores[tag] = 0 if self.correct_tags_number.get(tag, 0) == 0 else self.correct_tags_number.get(tag, 0) / self.pred_tags_count.get(tag)
        return precision_scores

    def cal_recall(self):
        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag,0) / self.gold_tags_count[tag]
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            f1_scores[tag] = 2*self.precision_scores[tag]*self.recall_scores[tag] / \
                                    (self.precision_scores[tag] + self.recall_scores[tag] + 1e-10)
        return f1_scores

    def count_correct_tags(self):
        correct_dict = {}
        for gold_tag, pred_tags in zip(self.golden_tags, self.pred_tags):
            if gold_tag == pred_tags:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1 
                else:
                    correct_dict[gold_tag] += 1 
        return correct_dict

    def remove_otags(self):
        length = len(self.golden_tags)
        o_tag_idx = [i for i in range(length) if self.golden_tags[i] == 'O']
        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags) if i not in o_tag_idx]
        self.pred_tags = [tag for i, tag in enumerate(self.pred_tags) if i not in o_tag_idx]

    def report_scores(self, dtype='HMM'):
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        with open('./result.txt', 'w') as f:
            f.write('\n')
            f.write('=========='*10)
            f.write('\n')
            f.write('模型：{}，test结果如下：'.format(dtype))
            f.write('\n')
            f.write(header_format.format('', *header))
            print(header_format.format('', *header))
            row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
            for tag in self.tagset:
                print(row_format.format(tag, self.precision_scores[tag], self.recall_scores[tag], self.f1_scores[tag], self.gold_tags_count[tag]))
                f.write('\n')
                f.write(row_format.format(tag, self.precision_scores[tag], self.recall_scores[tag], self.f1_scores[tag], self.gold_tags_count[tag]))

            avg_metrics = self._cal_weighted_average()
            print(row_format.format('avg/total', avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1_score'], len(self.golden_tags)))
            f.write('\n')
            f.write(row_format.format('avg/total', avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1_score'], len(self.golden_tags)))
            f.write('\n')
            
    def _cal_weighted_average(self):
        weighted_average = {}
        total = len(self.golden_tags)
        weighted_average['precision'] = 0
        weighted_average['recall'] = 0 
        weighted_average['f1_score'] = 0 
        for tag in self.tagset:
            size = self.gold_tags_count[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size
        for i in weighted_average.keys():
            weighted_average[i] /= total
        return weighted_average

    def report_confusion_metrix(self):
        print('\nConfusion Metrix')
        tag_list = list(self.tagset)
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)
        
        for gold_tag, pred_tag in zip(self.golden_tags, self.pred_tags):
            try:
                row = tag_list.index(gold_tag)
                col = tag_list.index(pred_tag)
                matrix[row][col] += 1 
            except:
                pass 

        row_format_ = '{:>7} ' * (tags_size+1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))
            



