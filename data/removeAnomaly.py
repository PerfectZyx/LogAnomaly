#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-input_seq', type=str, default='bgl.seq')
parser.add_argument('-input_label', type=str, default='bgl.label')
args = parser.parse_args()

normal_seq_file = open(args.input_seq+'_normal', 'w')
normal_index = 0
abnormal_index = 0
total_index = 0
with open(args.input_seq) as fin_seq:
    with open(args.input_label) as fin_label:
        for data, label in zip(fin_seq,fin_label):
            label = label.strip()
            total_index += 1
            if label == '1':
                abnormal_index += 1
            if label == '0':
                normal_index+=1
                normal_seq_file.writelines(data)
print('finished')
