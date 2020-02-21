# -*- coding: utf-8 -*-
# @Time    : 20/02/08 9:19
# @Author  : Wang Yu
# @Project : CHAT
# @File    : corpus_processing.py
# @Software: PyCharm


"""将王欣蕾组、刘书航组给定的原始数据正则、分词，存入excel后筛选"""


from MYTOOL import read_file, zhengze
import xlwings as xw
import jieba

RESULT_FILE = "data/TQR_ALL.xlsx"


# 预处理好的数据写入excel
def write2excel(file_name, sheet_name, *list_in):
    app = xw.App(visible=False, add_book=False)
    # try:
    wb = app.books.open(file_name)  # Book对象
    sht = wb.sheets.add(sheet_name)
    for i in range(len(list_in)):
        sht[0, i].options(transpose=True).value = list_in[i]
    wb.save(file_name)
    wb.close()
    app.quit()
    # except:
    #     app.quit()


# 处理LSHR.txt 正则化分词后放入excel
def task101_LSH_reply_processing():
    raw_r = read_file("data/0208/LSHR.txt", 0, 0)  # count 44729+1
    index_r = []
    r0 = []
    for line in raw_r[1:]:
        try:
            r = line.replace('"', '').strip().split('\t')
            index_r.append(int(r[1]))
            r0.append(' '.join(jieba.cut(zhengze(r[2].strip()))))
        except IndexError:
            pass
    # for i, j in zip(index_r, r0):
    #     print(i, j)
    write2excel(RESULT_FILE, 'LSHR', index_r, r0)


# 处理LSHQ.txt 正则化分词后放入excel
def task102_LSH_question_processing():
    raw_q = read_file("data/0208/LSHQ.txt", 0, 0)  # count 24294+1
    q0 = []  # 问题主题
    q1 = []  # 问题详细内容
    index_q = []
    for line in raw_q[1:]:
        try:
            q = line.replace('"', '').strip().split('\t')
            index_q.append(int(q[0]))
            q0.append(' '.join(jieba.cut(zhengze(q[1].strip()))))
            q1.append(' '.join(jieba.cut(zhengze(q[2].strip()))))
        except IndexError:
            pass
    # for i, j in zip(index_q, q0):
    #     print(i, j)
    write2excel(RESULT_FILE, 'LSHQ', index_q, q0, q1)


# 处理WXL1.txt 正则化分词后放入excel
def task103_WXL1_processing():
    raw_qr = read_file("data/0208/WXL1.txt", 0, 0)
    title = []
    que = []
    reply = []
    for line in raw_qr:
        try:
            tqr = line.strip().split('\t')
            title.append(' '.join(jieba.cut(zhengze(tqr[2].strip()))))
            que.append(' '.join(jieba.cut(zhengze(tqr[3].strip()))))
            reply.append(' '.join(jieba.cut(zhengze(tqr[4].strip()))))
        except IndexError:
            pass
    # for i,j,k in zip(title, que, reply):
    #     print(i)
    #     print(j)
    #     print(k)
    write2excel(RESULT_FILE, 'WXL1', title, que, reply)


# 处理WXL2.txt 正则化分词后放入excel
def task104_WXL2_processing():
    raw_qr = read_file("data/0208/WXL2.txt", 0, 0)
    title = []
    que = []
    reply = []
    for line in raw_qr:
        try:
            tqr = line.strip().split('\t')
            t0 = ' '.join(jieba.cut(zhengze(tqr[2].strip())))
            q0 = ' '.join(jieba.cut(zhengze(tqr[3].strip())))
            r0 = tqr[4].strip().split('|||')
            for i in r0:
                if len(i) > 0:
                    title.append(t0)
                    que.append(q0)
                    reply.append(' '.join(jieba.cut(zhengze(i))))
        except IndexError:
            pass
    # for i,j,k in zip(title, que, reply):
    #     print(i)
    #     print(j)
    #     print(k)
    write2excel(RESULT_FILE, 'WXL2', title, que, reply)


if __name__ == '__main__':
    # task101_LSH_reply_processing()      # 处理LSHR.txt 正则化分词后放入excel
    # task102_LSH_question_processing()   # 处理LSHQ.txt 正则化分词后放入excel
    # task103_WXL1_processing()           # 处理WXL1.txt 正则化分词后放入excel
    # task104_WXL2_processing()  # 处理WXL2.txt 正则化分词后放入excel

    # 问答语料库与处理完毕
    pass
