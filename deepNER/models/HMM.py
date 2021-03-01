#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   HMM.py
@Time    :   2021/02/23 17:47:10
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import torch

class HMM(object):
    def __init__(self, n, m):
        self.n = n  # len of target
        self.m = m  # len of vocab

        self.A = torch.zeros(n, n)
        self.B = torch.zeros(n, m)
        self.Pi = torch.zeros(n)

    def train(self, word_lists, tag_lists, word2id, tag2id):
        assert len(word_lists) == len(tag_lists)
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i+1]]
                self.A[current_tagid][next_tagid] += 1
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / torch.sum(self.A, dim=1, keepdim=True)

        for word_list, tag_list in zip(word_lists, tag_lists):
            assert len(word_list) == len(tag_list)
            for word, tag in zip(word_list, tag_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
        self.B[self.B==0.] = 1e-10
        self.B = self.B / torch.sum(self.B,dim=1,keepdim=True)

        for tag_list in tag_lists:
            init_tagId = tag2id[tag_list[0]]
            self.Pi[init_tagId] += 1
        self.Pi[self.Pi==0] = 1e-10
        self.Pi = self.Pi/self.Pi.sum()

    def decoding(self, word_list, word2id, tag2id):
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)
        seq_len = len(word_list)
        viterbi = torch.zeros(self.n, seq_len)
        backpointer = torch.zeros(self.n, seq_len).long()

        start_wordid = word2id.get(word_list[0], None)
        bt = B.t()
        if start_wordid is None:
            bt1 = (torch.ones(self.n)/self.n).long()
        else:
            bt1 = bt[start_wordid]
        viterbi[:,0] = Pi + bt1
        
        backpointer[:,0] = -1

        for step in range(1, seq_len):
            wordid = word2id.get(word_list[step], None)
            if wordid is None:
                bts = (torch.ones(self.n) / self.n).long()
            else:
                bts = bt[wordid]
            
            for tag_id in range(len(tag2id)):
                max_prob, max_id = torch.max(viterbi[:,step-1] + A[:,tag_id],dim=0)
                viterbi[tag_id,step] = max_prob + bts[tag_id]
                backpointer[tag_id, step] = max_id
        
        best_path_prob, best_path_pointer = torch.max(viterbi[:, seq_len - 1], dim=0)

        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len-1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)

        assert len(best_path) == len(word_list)
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]

        return tag_list

    def test(self, word_lists, word2id, tag2id):
        pred_tag_lists = []
        for word_list in word_lists:
            pred_tag_list = self.decoding(word_list, word2id, tag2id)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists

