# -*- coding: utf-8 -*-
# @Time    : 19/12/28 13:46
# @Author  : Wang Yu
# @File    : MYTOOL.py
# @Software: PyCharm


"""文本处理的自用工具"""

import re
import numpy as np
import jieba

# 读取文本文件
def read_file(file_name_in, lines_number_in=0, print_flag=False):
    """读取txt内容并打印输出前若干行"""
    with open(file_name_in, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
    if lines_number_in == 0:
        lines_number_in = len(lines1)

    if print_flag:
        for i in range(lines_number_in):
            print(lines1[i])
    lines_out = lines1[0:lines_number_in]
    return lines_out


# OK——正则表达式去除标点符号
def remove_biaodian(string_in, re_base="[^0-9A-Za-z\u4e00-\u9fa5]"):
    """正则表达式去除标点符号"""
    return re.sub(re_base, '', string_in)


# OK——正则表达式去除标点符号
def zhengze(string_in, re_base="[^，。？\u4e00-\u9fa5]"):
    """正则表达式去除标点符号"""
    return re.sub(re_base, '', string_in)


if __name__ == '__main__':
    data = read_file("data/1.txt", 0, 0)
    for line in data:
        a = line.strip().split('\t')
        print(' '.join(jieba.cut(a[0])),end='\t')
        print(' '.join(jieba.cut(a[1])),end='\t')
        print(' '.join(jieba.cut(a[2])))


