# -*- coding: utf-8 -*-
# @Time    : 20/02/09 16:36
# @Author  : Wang Yu
# @Project : CHAT
# @File    : Demo.py
# @Software: PyCharm


"""效果演示用"""


from Searching import *
from Generating import *


# 进行系统检查并打印检查结果
def system_check():
    return Searching_check(True) and Generating_check(True)


# 通过系统检查后开始演示案例
def demo():
    if system_check():
        print(initial_jieba("### 模型初始化中，请稍后 ###"))
        model, title, question, answer, q_matrix, fdist, singular_v = Searching_load()
        searcher, voc = Generating_load()
        while True:
            q_in = input('>:')
            if q_in.lower() == "q":
                break

            S_result = top5_question(que2vec(q_in, model, fdist, singular_v, 'SIF'), q_matrix)
            credit = S_result[0][1]
            S_reply = answer[S_result[0][0]].replace(" ", "")
            G_reply = evaluate_input(searcher, voc, q_in)
            if credit >= 0.8:
                print("-:", S_reply)
            else:
                print("-:", G_reply)


if __name__ == '__main__':
    demo()
