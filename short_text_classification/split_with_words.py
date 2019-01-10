#encoding:utf-8
import io
import re
import sys
import string
import argparse
import jieba
from zhon.hanzi import punctuation

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('src_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

with open(args['out_path'], 'w') as f:
    for line in io.open(args['src_path'], 'r', encoding='utf-8'):
        parts = line.strip().split("\t")

        parts[1] = parts[1].replace("&amp;quot;", "")
        parts[1] = parts[1].replace(u"\u200b", "")
        parts[1] = re.sub("\s+", "", parts[1])
        parts[1] = re.sub("\d+", "", parts[1])
        # 去除所有中文标点符号
        parts[1] = re.sub(ur"[%s]+" % punctuation, "", parts[1])
        # 去除所有英文标点符号
        parts[1] = re.sub(ur"[%s]+" % string.punctuation, "", parts[1])

        #seg_list = jieba.cut(parts[1], cut_all=True)
        #seg_list = jieba.cut(parts[1], cut_all=False)
        seg_list = [word for word in parts[1]]
        seg_words = " ".join(seg_list)
        f.write(parts[0].encode('utf-8') + "\t" + seg_words.encode('utf-8') + '\n')

