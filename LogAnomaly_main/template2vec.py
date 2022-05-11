#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np


class Template2Vec:
    def __init__(self, model_file, template_file, is_binary=False):
        """ 初始化函数
        
        """
        print('reading template2vec model')
        model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary = is_binary)
        template_to_index = {}  # index从0开始(用于lstm)，template(模板编号)从1开始
        index_to_template = {}
        
        template_num = 0  # 模板数
        with open(template_file) as fin:
            for line in fin:
                template_num += 1
        
        template_to_index = dict((str(i+1), i) for i in range(template_num))
        index_to_template = dict((i, str(i+1)) for i in range(template_num))
        
        template_matrix = []
        for i in range(template_num):
            key = str(i+1)
            template_matrix.append(model[key])
        self.template_matrix = np.mat(template_matrix)
        
        vector_template_tuple =[(model[key], key) for key in template_to_index]  # 向量与模板编号的映射关系
        self.model = model
        self.template_to_index = template_to_index
        self.index_to_template = index_to_template
        self.template_num = len(template_to_index)
        self.dimension = len(model['1'])
        self.vector_template_tuple = vector_template_tuple
        print(' Template2Vec.dimension:', self.dimension)
        print(' Template2Vec.template_num:', self.template_num)

    def word_to_most_similar(self, in_word, topn = 1):
        ''' 与word最相似的
        Args: 
            word
        Return: 
            tuple(template_index,similarity)
        '''
        index = self.model.most_similar(positive = in_word,topn = topn)
        return index 
    
    def vector_to_most_similar(self, in_vector, topn = 1):
        ''' 与vector最相似的word
        Args: 
            vector
        Return: 
            与vector最相似的word，包含其本身。top1应该是vector对应的word。
        '''
        temp_dict = {}
        for t in self.vector_template_tuple:
            template_index = t[1]
            vector = t[0]
            temp_dict[template_index] = self.cos(in_vector, vector)
        sorted_final_tuple=sorted(temp_dict.items(),key=lambda asd:asd[1] ,reverse=True)
        return sorted_final_tuple[:topn] 
    
    def cos(self, vector1, vector2):
        ''' 计算两个vector的cos
        Args: 
            vector
        Return：
            cos
        '''
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        x = float(np.sum(vector1 * vector2)) / norm1 * norm2
        return x 
        
        
if __name__ == '__main__':
    temp2Vec_file = '../../model/bgl_log.template_vector'
    template_file = '../../middle/bgl_log.template'
    t = Template2Vec(temp2Vec_file, template_file)

    print(t.word_to_most_similar(['26'],topn = 3))
    print(t.vector_to_most_similar(t.model['26'],topn = 4))
    print(t.vector_to_most_similar(t.model['26'],topn = 1)[0][0])
    print(t.vector_to_most_similar(t.model['26'],topn = 4))
