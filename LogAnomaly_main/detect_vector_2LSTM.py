#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.metrics import precision_recall_fscore_support
import numpy
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras import backend as K
import argparse
import os
from template2vec import Template2Vec


def find_newest_file(dir_path):
    ''' 找到指定目录下的最新文件
    Args: 
        dir_path: 目录
    Return:
        newest_file: 最新文件
    '''
    filenames = os.listdir(dir_path)
    name_ = []
    time_ = []
    for filename in filenames:
        if 'DS' not in  filename and 'hdf5' in filename:
            c_time = os.path.getctime(dir_path+filename)
            name_.append(dir_path+filename)
            time_.append(c_time)
    newest_file = name_[time_.index(max(time_))]
    return newest_file


def detect_anomaly(para):
    ''' 异常检测
    Args: 
        para: 参数
    Return:
    '''
    import time
    t1=time.time()

    filename = para['test_file']
    seq_length = para['seq_length']
    n_candidates = para['n_candidates']  # topn候选集
    windows_size = para['windows_size']  # 时间窗口大小
    step_size = para['step_size']  # 时间窗口的滑动步长
    onehot = para['onehot']  # 1表示统计使用onehot，0表示使用template2vec
    model_filename = para['model_filename']  # 训练的模型参数
    model_dir = para['model_dir']  # 模板数量
    template_index_map_path = para['template_index_map_path']  # 保存模板号与向量的映射关系
    result_file = para['result_file']
    template_num = para['template_num']
    label_file = para['label_file']
    template2Vec_file = para['template2Vec_file']
    tempalte_file = para['template_file']
    count_matrix_flag = para['count_matrix']
    temp2Vec = Template2Vec(template2Vec_file, tempalte_file)

    #如果没有指定model_filename, 则从weight/文件夹中找出最新生成的文件
    if model_filename == '':
        model_filename = find_newest_file(model_dir)
        print('cur_model_filename',model_filename)


    template_to_int = {}
    int_to_template = {}
    if template_num == 0:
        # 如果template_num为0，则根据模板序列文件来生成映射
        with open(template_index_map_path) as IN:
            for line in IN:
                l = line.strip().split()
                c = l[0]
                i = int(l[1])
                template_to_int[c] = i
                int_to_template[i] = c
    else:
        # 如果template_num不为0，则根据其构造映射,int从0开始，char从1开始
        template_to_int = dict((str(i+1), i) for i in range(template_num))
        int_to_template = dict((i, str(i+1)) for i in range(template_num))

    raw_text = []
    raw_time_list = []
    raw_label_list = []
    with open(filename) as line_IN:
        with open(label_file) as label_IN:
            for line, label_line in zip(line_IN, label_IN):
                l=line.strip().split()
                if l[1] != '-1' and l[1] !='0' and l[1] in template_to_int:
                    raw_text.append(l[1])
                    raw_label_list.append(int(label_line.strip()))

    chars = sorted(list(set(raw_text)))

    n_chars = len(raw_text)
    n_templates = len(template_to_int)
    print ("length of log sequence: ", n_chars)
    print ("# of templates: ", n_templates)

    charX = []
    label_list = []
    vectorX = []
    vectorY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        label_out = raw_label_list[i + seq_length]
        charX.append(seq_in)
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
        vectorX.append(temp_list)
        vectorY.append(temp2Vec.model[seq_out])
        label_list.append(label_out)
    n_patterns = len(vectorX)
    print ("# of patterns: ", n_patterns)

    if count_matrix_flag == 0:
        X = numpy.reshape(vectorX, ( -1, seq_length, temp2Vec.dimension)) #
    else:
        X = numpy.reshape(vectorX, ( -1, seq_length, temp2Vec.dimension + n_templates))
    y = numpy.reshape(vectorY,(-1,temp2Vec.dimension))

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

    # 加载网络权重
    model.load_weights(model_filename)
    model.compile(loss='mse', optimizer='adam')
    if onehot ==1:
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    total = 0
    anomaly_count_dir = {}
    for i in range(n_candidates):
        anomaly_count_dir[i+1] = []
    test1_time = time.time()
    for x_char,x,aim_y_vector in zip(charX, X, y):
        total += 1
        if total % 1000 ==0:
            test2_time = time.time()
            print(str(total)+'/'+str(len(X)),str( round(100*total/len(X),3) ),'% time:',(test2_time - test1_time)/60)
            test1_time = time.time()
        aim_y_char = temp2Vec.vector_to_most_similar(aim_y_vector, topn = 1)[0][0]
        if count_matrix_flag == 0:
            x = numpy.reshape(x, (1, seq_length, temp2Vec.dimension))
        else:
            x = numpy.reshape(x, (1, seq_length, temp2Vec.dimension + n_templates))
        prediction = model.predict([x[:,:,:temp2Vec.dimension],x[:,:,temp2Vec.dimension:]], verbose=0)[0] #输出一个len(tags)的向量，数值越高的列对应概率最高的类别

        #获取最相似的topn
        if onehot == 1:
            for i in range(n_candidates):
                i += 1
                top_n_index = prediction.argsort()[-i:]
                top_n_tag=[int_to_template[index] for index in top_n_index]
                if aim_y_char not in top_n_tag:
                    anomaly_count_dir[i].append(1)
                else:
                    anomaly_count_dir[i].append(0)

        else:
            top_n_tuple = temp2Vec.vector_to_most_similar(prediction, topn=n_candidates)
            for i in range(n_candidates):
                i += 1
                top_n =[t[0] for t in top_n_tuple[:i]]
                if aim_y_char not in top_n:
                    anomaly_count_dir[i].append(1)
                else:
                    anomaly_count_dir[i].append(0)

    f = open(result_file,'w')

    print('\nanomaly detection result:')
    for i in range(n_candidates):
        i += 1
        print('next tag  is not in top'+str(i)+' candidates:')
        precision, recall, f1_score, _ = numpy.array(list(precision_recall_fscore_support(label_list, anomaly_count_dir[i])))[:, 1]
        print('=' * 20, 'RESULT', '=' * 20)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for ground_truth, detected, in zip(label_list, anomaly_count_dir[i]):
            if ground_truth == 1 and detected == 1:
                tp += 1
            if ground_truth == 1 and detected ==0:
                fn += 1
            if ground_truth ==0 and detected == 0:
                tn += 1
            if ground_truth ==0 and detected == 1:
                fp += 1
        print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))
        print('tp:',tp, 'fn:',fn,'tn:',tn,'fp:',fp,'total:',tp+tn+fp+fn)
        print('=' * 20, 'RESULT', '=' * 20)
        f.writelines(str(precision)+' '+str(recall)+'\n')

    f.close()
    t2 = time.time()
    print('testing time:',(t2-t1)/60,'mins')
    print ("\nDone.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_file', help='test_file.', type=str, default='../../middle/bgl_log.seq')
    parser.add_argument('-seq_length', help='seq_length.', type=int, default=10)
    parser.add_argument('-n_candidates', help='n_candidates.', type=int, default=15)
    parser.add_argument('-windows_size', help='windows_size.', type=int, default=3)
    parser.add_argument('-step_size', help='step_size.', type=int, default=1)
    parser.add_argument('-model_filename', help='you can give a model file.', type=str, default='')
    parser.add_argument('-model_dir', help='model_dir.', type=str, default='../weights/vector_deeplog/')
    parser.add_argument('-template_index_map_path', help='template_index_map_path.', type=str, default='./bgl_log_template_to_int.txt')
    parser.add_argument('-onehot', help='默认为1。1表示统计使用onehot，0表示使用template2vec',type = int, default = 1)
    parser.add_argument('-result_file', help='result_file.', type=str, default='../results/bgl_log_log_pr.txt')
    parser.add_argument('-template_num', help='若为0，则根据输入文件统计，否则，根据输入确定。默认0', type=int, default=0)
    parser.add_argument('-label_file', help='label_file.', type=str, default='../../data/bgl_label')
    parser.add_argument('-count_matrix', help='默认为0。1表示统计count_matrix，0不统计',type = int, default = 0)
    parser.add_argument('-template2Vec_file', help='template2Vec_file', type=str, default='../../model/bgl_log.template_vector')
    parser.add_argument('-template_file', help='template_file', type=str, default='../../middle/bgl_log.template')

    args = parser.parse_args()

    para_detect = {
        'test_file': args.test_file,
        'seq_length':args.seq_length,
        'n_candidates': args.n_candidates,
        'windows_size': args.windows_size,
        'step_size':args.step_size,
        'model_dir': args.model_dir,
        'model_filename': args.model_filename,
        'template_index_map_path':args.template_index_map_path,
        'template_num' : args.template_num,
        'result_file':args.result_file,
        'label_file':args.label_file,
        'template2Vec_file': args.template2Vec_file,
        'template_file': args.template_file,
        'count_matrix': args.count_matrix,
        'onehot': args.onehot
        }

    detect_anomaly(para_detect)
    print('detection finish')

    K.clear_session()
