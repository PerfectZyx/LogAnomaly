#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np


def load_model(filename,is_binary=False):
    ''' 加载模型
    Args: 
        filename: 文件
    Return:
        model: 模型
    '''
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary = is_binary)
    return model

def getTemplateVector(para):
    ''' 通过模板文件中的单词词向量，组合成每条日志模板的句向量，然后将模板向量保存到文件中，每个模板向量的index是模板号，从1开始
    Args:
        para: 参数
    Returns:
        (template_to_index, index_to_template, template_to_vector)    
    '''
    template_file = para['template_file']
    model = load_model(para['word_model'])
    dimension = para['dimension']
    template_vector_file = para['template_vector_file']
    template_to_index = {}
    index_to_template = {}
    template_to_vector = {}
    template_num = 0
    with open(template_file) as IN:
        for line in IN:
                template_num += 1
    f = open(template_vector_file, 'w')
    f.writelines(str(template_num)+' '+str(para['dimension'])+'\n') #word2vec的模型格式，第一行为单词数&维度
    index = 1
    with open(template_file) as IN:
        for line in IN:
            template = line.strip()
            l = template.split()
            cur_vector = np.zeros(dimension)
            for word in l:
                cur_vector += model[word]
            cur_vector /= len(l)
            template_to_vector[template] = cur_vector
            template_to_index[template] = str(index)
            index_to_template[index] = template
            f.writelines(str(index))
            for v in cur_vector:
                f.writelines(' '+str(v))
            f.writelines('\n')
            index += 1
    return (template_to_index, index_to_template, template_to_vector)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-template_file', help='template_file', type=str, default='../../middle/M2_template.txt')
    parser.add_argument('-word_model', help='word_model', type=str, default='../../model/M2.model')
    parser.add_argument('-template_vector_file', help='template_vector_file', type=str, default='../../model/M2_template.vector')
    parser.add_argument('-dimension', help='dimension', type=int, default=200)
    args = parser.parse_args()

    para = {
        'template_file' : args.template_file,
        'word_model': args.word_model,
        'template_vector_file': args.template_vector_file,
        'dimension':args.dimension
    }
    getTemplateVector(para)
    print('end~~')
