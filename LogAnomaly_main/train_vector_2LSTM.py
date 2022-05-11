#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import argparse
from template2vec import Template2Vec
import time


def create_dir(path):
    ''' 创建目录
    Args: 
        path: 目录
    Return:
    '''
    import os
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def train_model(para):
    ''' 训练模型
    Args: 
        para: 参数
    Return:
    '''
    t1 =time.time()
    epoch = para['epoch']
    filename = para['train_file']
    seq_length = para['seq_length']
    model_dir = para['model_dir']
    template_num = para['template_num']
    template2Vec_file = para['template2Vec_file']
    tempalte_file = para['template_file']
    count_matrix_flag = para['count_matrix']
    onehot = para['onehot']
    temp2Vec = Template2Vec(template2Vec_file, tempalte_file)


    create_dir(model_dir)
    template_index_map_path = para['template_index_map_path']  # 保存模板号与向量的映射关系
    raw_text = []
    with open(filename) as fin:
        for line in fin:
            l=line.strip().split()
            if l[1] != '-1':
                raw_text.append(l[1])
    t_read_raw_log = time.time()
    print('t_read_raw_log',(t_read_raw_log-t1)/60,'mins')

    if template_num == 0:
        # 如果template_num为0，则根据模板序列文件来生成映射
        chars = sorted(list(set(raw_text)))
        template_to_int = dict((c, i) for i, c in enumerate(chars))
        print('template_to_int', template_to_int)
        f = open(template_index_map_path,'w')
        for k in template_to_int:
            f.writelines(str(k)+' '+str(template_to_int[k])+'\n')
        f.close()
    else:
        # 如果template_num不为0，则根据其构造映射,int从0开始，char从1开始
        template_to_int = dict((str(i+1), i) for i in range(template_num))
        print('template_to_int', template_to_int)

    n_chars = len(raw_text)
    n_templates = len(template_to_int)
    print ("length of log sequence: ", n_chars)
    print ("# of templates: ", n_templates)

    dataX = []
    dataY = []
    vectorX = []
    vectorY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataY.append(template_to_int[seq_out])
        temp_list = []
        for seq in seq_in:
            if count_matrix_flag == 0:
                #不拼接，直接用template vector
                temp_list.append(list(temp2Vec.model[seq]))
            else:
                #拼接template vector和count vector
                cur_count_vector = [0 for i in range(n_templates)]
                for t in seq_in:
                    cur_index = template_to_int[t]
                    cur_count_vector[cur_index]+=1

                l =list(temp2Vec.model[seq])
                l.extend(cur_count_vector)
                temp_list.append(l)

        vectorX.append(numpy.array(temp_list))
        vectorY.append(temp2Vec.model[seq_out])

    n_patterns = len(vectorX)
    print ("# of patterns:", n_patterns)
    t_generate_vector = time.time()
    print('generateVector time:',(t_generate_vector - t_read_raw_log)/60,'mins' )

    if count_matrix_flag == 0:
        X = numpy.reshape(vectorX, ( -1, seq_length, temp2Vec.dimension))
    else:
        X = numpy.reshape(vectorX, ( -1, seq_length, temp2Vec.dimension + n_templates))
    y = numpy.reshape(vectorY,(-1,temp2Vec.dimension))
    t_reshape = time.time()
    print('t_reshape:',(t_reshape - t_generate_vector)/60,'mins' )

    if onehot ==1:
        y = to_categorical(dataY, num_classes = n_templates)
    t_tocategorical = time.time()
    print('t_tocategorical:',(t_tocategorical - t_reshape)/60,'mins' )


    model_vector_input = Input(shape=(X.shape[1], temp2Vec.dimension))
    model_vector_hidden = LSTM(128, input_shape=(X.shape[1], n_templates), return_sequences=False)(model_vector_input)
    model_vector_output = Dropout(0.2)(model_vector_hidden)
    model_vec = Model(model_vector_input, model_vector_output)

    model_count_input = Input(shape=(X.shape[1], n_templates))
    model_count_hidden = LSTM(128, input_shape=(X.shape[1], n_templates), return_sequences=False)(model_count_input)
    model_count_output = Dropout(0.2)(model_count_hidden)
    model_count = Model(model_count_input, model_count_output)
    
    concatenated = concatenate([model_vector_output, model_count_output])
    if onehot == 0:
        out = Dense(temp2Vec.dimension, activation='softmax')(concatenated)
    else:
        out = Dense(n_templates, activation='softmax')(concatenated)
    
    model = Model([model_vector_input, model_count_input], out)
    model.compile(loss='mse', optimizer='adam')
    if onehot ==1:
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    if count_matrix_flag == 0:
        s = 'only_vector'
    else:
        s = 'contact_matrix'
    filepath = model_dir+"log_weights-"+ s +"-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit([X[:,:,:temp2Vec.dimension], X[:,:,temp2Vec.dimension:]], y, batch_size=64, epochs=epoch, callbacks=callbacks_list)
    
    t2 = time.time()
    print('training time:',(t2-t1)/60,'mins')
    return n_templates


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', help='train_file.', type=str, default='../../middle/bgl_log.seq')
    parser.add_argument('-seq_length', help='seq_length.', type=int, default=10)
    parser.add_argument('-model_dir', help='网络参数的输出文件夹', type=str, default='../weights/vector_deeplog/')
    parser.add_argument('-template_num', help='若为0，则根据输入文件统计，否则，根据输入确定。默认0', type=int, default=0)
    parser.add_argument('-template2Vec_file', help='template2Vec_file', type=str, default='../../model/bgl_log.template_vector')
    parser.add_argument('-count_matrix', help='默认为0。1表示统计count_matrix，0不统计',type = int, default = 0)
    parser.add_argument('-onehot', help='默认为1。1表示统计使用onehot，0表示使用template2vec',type = int, default = 1)
    parser.add_argument('-template_file', help='template_file', type=str, default='../../middle/bgl_log.template')
    parser.add_argument('-epoch', help='epoch', type=int, default=30)
    args = parser.parse_args()

    para_train = {
        'train_file': args.train_file,
        'seq_length':args.seq_length,
        'model_dir': args.model_dir,
        'template_index_map_path':args.train_file+'_map',
        'template_num': args.template_num,
        'template2Vec_file': args.template2Vec_file,
        'template_file': args.template_file,
        'count_matrix': args.count_matrix,
        'onehot': args.onehot,
        'epoch': args.epoch
        }

    n_templates = train_model(para_train)
    
    K.clear_session()
    print('training has finished')
