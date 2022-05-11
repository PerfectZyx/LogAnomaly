#!/usr/bin/python
# -*- coding: UTF-8 -*-

from nltk.corpus import wordnet
from itertools import chain
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO,filename='wd.log')

def build_dictionary_wordnet(template_file, syn_file, ant_file):
    """ 根据wordnet构建字典
    Args:
        template_file: 模板文件
        syn_file: 同义词文件
        ant_file: 反义词文件
    Returns:
    """
    synonyms = {}
    antonyms = {}
    vocabulary = set()

    with open(template_file) as fin:
        for line in fin:
            l = line.strip().split()
            for word in l:
                vocabulary.add(word)

    for word in vocabulary:
            synsets = wordnet.synsets(word)
            if len(synsets) == 0:
                logging.info('Not find: %s'%word)
                continue
            synonyms[word] = set(chain.from_iterable([syn.lemma_names() for syn in synsets]))
            antsets = set()
            for syn in synsets:
                for lem in syn.lemmas():
                    if lem.antonyms():
                        antsets.add(lem.antonyms()[0].name())
            if len(antsets) == 0:
                logging.info('No antonyms: %s'%word)
                continue
            antonyms[word] = antsets

    fout = open(syn_file,'w')
    for key in sorted(synonyms.keys()):
        for syn in synonyms[key]:
            if syn in vocabulary and key in vocabulary:
                fout.write('%s\t%s\n'%(key,syn))
    fout.close()
    fout = open(ant_file,'w')
    for key in sorted(antonyms.keys()):
        for ant in antonyms[key]:
            if ant in vocabulary and key in vocabulary:
                fout.write('%s\t%s\n'%(key,ant))
    fout.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', help = 'data_prefix', type = str, default = '../../../middle/')
    parser.add_argument('-template_file', help = 'template_file', type = str, default = 'bgl_template.txt')
    parser.add_argument('-syn_file',help = 'syn_file', type = str, default = 'bgl_synonyms.txt')
    parser.add_argument('-ant_file',help = 'ant_file', type = str, default = 'bgl_antonyms.txt')
    args = parser.parse_args()

    template_file = args.data_dir + args.template_file
    syn_file = args.data_dir + args.syn_file
    ant_file = args.data_dir + args.ant_file

    build_dictionary_wordnet(template_file, syn_file, ant_file)
    print('end~~')
