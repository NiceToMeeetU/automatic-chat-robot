# 基于生成式+检索式的自动问系统

安装需要的环境，具体环境要求详见文件
基本都是常见的第三方模块，jieba, numpu, sklearn, nltk, gensim, sklearn, pickle, torch
环境配置完成后直接运行Demo.py即可，该演示文件直接在控制台输入输出，你们可以根据情况改成与后台对接


文件说明：
corpus_processing.py	对原始数据的处理
Demo.py			演示主程序
Generating.py		生成式方法
MYTOOL.py		公用的部分文本处理函数
Searching.py		检索式方法


数据说明：
corpus.txt		实际筛选出的问答语料库
model.model		词嵌入模型
NN-model.tar		循环网络模型
QD.txt		问题向量矩阵

