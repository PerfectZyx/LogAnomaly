## 环境：
	python=3.8
    tensorflow==2.5   


## 模板提取：
* 运行脚本的的命令：
    * cd LogAnomaly/ft_tree/
    * python -u ft_tree.py -data_path ../data/bgl.log -template_path ../middle/bgl_log.template -fre_word_path ../middle/bgl_log.fre
* 参数样例：
    * data_path：日志文件 ../data/bgl.log
    * template_path：模板文件 ../middle/bgl_log.template
    * fre_word_path：单词词频文件 ../middle/bgl_log.fre

## 模板匹配：
* 运行脚本的的命令：
    * cd LogAnomaly/ft_tree/
    * python -u matchTemplate.py -template_path ../middle/bgl_log.template -fre_word_path ../middle/bgl_log.fre -log_path ../data/bgl.log -out_seq_path ../data/bgl_log.seq -match_model 1
* 参数样例：
    * template_path：模板文件 ../middle/bgl_log.template
    * fre_word_path：单词词频文件 ../middle/bgl_log.fre
    * log_path：日志文件 ../data/bgl.log
    * out_seq_path：日志序列文件 ../data/bgl_log.seq
    * match_model：1:正常匹配日志  2:单条增量学习&匹配 3:批量增量学习&匹配

## 获取正常日志序列：
* 运行脚本的的命令：
    * cd LogAnomaly/data/
    * python removeAnomaly.py -input_seq bgl_log.seq -input_label bgl.label
* 参数样例：
    * input_seq：日志序列文件 bgl_log.seq
    * input_label：日志label文件 bgl.label


## 基于模板搜索同义词和反义词：
* 运行脚本的的命令：
    * cd LogAnomaly/template2Vector/src/preprocess/
    * python wordnet_process.py -data_dir ../../../middle/ -template_file bgl_log.template -syn_file bgl_log.syn -ant_file bgl_log.ant
* 参数样例：
    * data_dir：数据目录 ../../../middle/
    * template_file：模板文件 bgl_log.template
    * syn_file：同义词文件 bgl_log.syn
    * ant_file：反义词文件 bgl_log.ant

## 转换日志模板为词向量训练格式：
* 运行脚本的的命令：
    * cd LogAnomaly/middle/
    * python changeTemplateFormat.py -input bgl_log.template
* 参数样例：
    * input：模板文件 bgl_log.template

## 日志模板学习词向量：
* 运行脚本的的命令：
    * cd LogAnomaly/template2Vector/src/
    * make
    * ./lrcwe -train ../../middle/bgl_log.template_for_training -synonym ../../middle/bgl_log.syn -antonym ../../middle/bgl_log.ant -output ../../model/bgl_log.model -save-vocab ../../middle/bgl_log.vector_vocab -belta-rel 0.8 -alpha-rel 0.01 -belta-syn 0.4 -alpha-syn 0.2 -alpha-ant 0.3 -size 32 -min-count 1
* 参数样例：
    * train：训练文件 ../../middle/bgl_log.template_for_training
    * synonym：同义词文件 ../../middle/bgl_log.syn
    * antonym：反义词文件 ../../middle/bgl_log.ant
    * output：单词模型 ../../model/bgl_log.model
    * save-vocab：保存单词向量 ../../middle/bgl_log.vector_vocab
    * belta-rel 0.8 
    * alpha-rel 0.01 
    * belta-syn 0.4 
    * alpha-syn 0.2 
    * alpha-ant 0.3 
    * size 32 
    * min-count：最小词频训练阀值 1

## 获取模板向量：
* 运行脚本的的命令：
    * cd LogAnomaly/template2Vector/src/
    * python template2Vec.py -template_file ../../middle/bgl_log.template -word_model ../../model/bgl_log.model -template_vector_file ../../model/bgl_log.template_vector -dimension 32
* 参数样例：
    * template_file：模板文件 ../../middle/bgl_log.template
    * word_model：单词模型 ../../model/bgl_log.model
    * template_vector_file：模板向量文件 ../../model/bgl_log.template_vector
    * dimension：维度 32

## 模型训练：
* 运行脚本的的命令：
    * cd LogAnomaly/LogAnomaly_main/
    * python -u train_vector_2LSTM.py -train_file ../data/bgl_log.seq_normal -seq_length 10 -model_dir ../weights/vector_matrix/ -onehot 1 -template2Vec_file ../model/bgl_log.template_vector -template_file ../middle/bgl_log.template -count_matrix 1
* 参数样例：
    * train_file：训练文件 ../data/bgl_log.seq_normal
    * seq_length：序列长度 10
    * model_dir：模型目录 ../weights/vector_matrix/
    * onehot：1:使用独热编码  0:不使用独热编码  1
    * template2Vec_file：模板向量文件 ../model/bgl_log.template_vector
    * template_file：模板文件 ../middle/bgl_log.template
    * count_matrix：1:统计count_matrix  0:不统计  1

## 异常检测：
* 运行脚本的的命令：
    * cd LogAnomaly/LogAnomaly_main/
    * python -u detect_vector_2LSTM.py -test_file ../data/bgl_log.seq -seq_length 10 -model_dir ../weights/vector_matrix/ -n_candidates 15 -windows_size 3 -step_size 1 -onehot 1 -result_file ../results/bgl_log_precision_recall.txt -label_file ../data/bgl.label -template2Vec_file ../model/bgl_log.template_vector -template_file ../middle/bgl_log.template -count_matrix 1
* 参数样例：
    * test_file：测试文件 ../data/bgl_log.seq
    * seq_length：序列长度 10
    * model_dir：模型目录 ../weights/vector_matrix/
    * n_candidates：日志候选集数目 15
    * windows_size：时间窗口大小 3
    * step_size：时间窗口步长 1
    * onehot：1:使用独热编码  0:不使用独热编码  1
    * result_file：模型评估结果输出 ../results/bgl_log_precision_recall.txt
    * label_file：日志label文件 ../data/bgl.label
    * template2Vec_file：模板向量文件 ../model/bgl_log.template_vector
    * template_file：模板文件 ../middle/bgl_log.template
    * count_matrix：1:统计count_matrix  0:不统计  1


## 注意
1. 由于word2vec是已有方法，所用到的word2vec相关代码均直接使用Google提供的源码。
2. 由于本文重点在于异常检测，日志解析用到的是当前已有算法，所以用到的日志解析方法FT-tree的代码直接使用作者论文源码。
3. 改进部分在LogAnomaly/integration中。
