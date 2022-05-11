#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-input',help='input file name ',type = str, default ='bgl_log.template')
arg = parser.parse_args()
in_file = arg.input
out_file = input+'_for_training'

f = open(out_file,'w')
with open(in_file) as fin:
    for line in fin:
        f.writelines(line.strip()+' ')
f.writelines('\n')

print('input:',in_file,'output:',out_file)
