# coding   : utf-8 
# @Time    : 20/11/06 16:25
# @Author  : Wang Yu
# @Project : automatic-chat-robot
# @File    : Searching.py
# @Software: PyCharm


"""检索式回答机制"""


from MYTOOL import read_file, zhengze
import jieba
from gensim.models import word2vec
import numpy as np
import nltk
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os.path

CORPUS_FILE = './data/corpus.txt'
MODEL_FILE = './data/model.model'
MATRIX_FILE = './data/QD.txt'


# 训练词向量
def corpus_training():
    sentences_in = word2vec.Text8Corpus(CORPUS_FILE)
    model = word2vec.Word2Vec(sentences_in, size=150, window=5, min_count=1, workers=4)
    model.save(MODEL_FILE)
    print("w2v model trained successfully.")


# 拼接词向量，构成长句向量
def s2v(model_in, sentence_in):
    vec = []
    for word in sentence_in:
        try:
            tmp = model_in.wv[word]
        except KeyError:
            tmp = [0 for _ in range(150)]
        vec.append(tmp)
    vec = np.array(vec)
    return vec


# SIF 法构建问题向量，进而拼接总的问题矩阵
def question_matrix(model_in, corpus_in, pattern='SIF', sif_weight=0.0001):
    if pattern == 'AVG':
        Q = []
        for query in corpus_in:
            tmp_vec = np.array(s2v(model_in, query))
            Q_vec = tmp_vec.mean(axis=0)
            Q.append(Q_vec)
        Q_matrix = np.array(Q)
        return Q_matrix, 0, 0
    elif pattern == 'SIF':
        Q = []
        raw = []
        merge = []
        weight = []
        for i in range(len(corpus_in)):
            merge.extend(corpus_in[i])
        fdist = nltk.probability.FreqDist(merge)
        # count = 0
        for query in corpus_in:
            tmp_vec = np.array(s2v(model_in, query))
            weight_matrix = np.array([[sif_weight / (sif_weight + fdist[word] / fdist.N())
                                       for word in query]])
            tmp = tmp_vec * weight_matrix.T
            Q_vec = tmp.mean(axis=0)
            Q.append(Q_vec)
            weight.append(weight_matrix)
            raw.append(tmp_vec)
        Q_matrix = np.array(Q)
        pca = PCA(n_components=1)
        u = pca.fit_transform(Q_matrix.T)
        res = Q_matrix - np.dot(Q_matrix, np.dot(u, u.T))
        print("Question matrix calculated successfully.")
        return res, fdist, u


# 构建问题库
class QuestionDatabase:
    def __init__(self, q_matrix, fdist, singular_v):
        self.q_matrix = q_matrix
        self.fdist = fdist
        self.singular_v = singular_v


# 加载问题库
def load_QD(file_name):
    fr = open(file_name, 'rb')
    QD = pickle.load(fr)
    return QD.q_matrix, QD.fdist, QD.singular_v


# 新输入问题转向量
def que2vec(query, model_in, fdist, singular_v, pattern='SIF', sif_weight=0.0001):
    query = list(jieba.cut(zhengze(query.strip())))
    # query_vec = np.zeros(100)
    if pattern == 'AVG':
        vec = np.array(s2v(model_in, query))
        query_vec = vec.mean(axis=0)
    elif pattern == 'SIF':
        vec = np.array(s2v(model_in, query))
        try:
            pass
        except ValueError:
            pass

        weight = np.array([[sif_weight / (sif_weight + fdist[word] / fdist.N())
                            for word in query]])
        tmp = vec * weight.T
        tmp_vec = tmp.mean(axis=0)
        query_vec = tmp_vec - np.dot(np.dot(singular_v, singular_v.T), tmp_vec)

    return query_vec


# 计算近似度
def cal_similarity(query_vec, q_matrix):
    sim_dict = {}
    for i in range(q_matrix.shape[0]):
        sim = np.dot(query_vec, q_matrix[i].T) / (np.linalg.norm(query_vec) * np.linalg.norm(q_matrix[i]))
        if sim > 0:
            sim_dict[i] = sim
    return sim_dict


# 粗暴计算余弦近似值，用于检验，弃用
def similarity(query_vec, q_matrix):
    sim_dict = {}
    query_vec = query_vec.reshape(1, 100)
    for i in range(q_matrix.shape[0]):
        try:
            sim = cosine_similarity(query_vec, q_matrix[i].reshape(1, 100))[0][0]
            if sim > 0:
                sim_dict[i] = sim
        except ValueError:
            pass
    return sim_dict


# 找出前5大回答
def top5_question(query_vec, q_matrix):
    sim_dic = cal_similarity(query_vec, q_matrix)
    # sim_dic = similarity(query_vec, q_matrix)
    # d = sorted(sim_dic, key=lambda x: sim_dic[x], reverse=True)
    top5 = sorted(sim_dic.items(), key=lambda d: d[1], reverse=True)[:5]
    return top5


# 读取完整版数据
def TQR_import():
    raw_data = read_file(CORPUS_FILE, 0, 0)
    title = []
    question = []
    reply = []
    for line in raw_data:
        tqr = line.strip().split('\t')
        title.append(tqr[0].split(' '))
        question.append(tqr[1].split(' '))
        reply.append(tqr[2])
    return title, question, reply


# 检查各种文件是否存在
def Searching_check(print_flag):
    check_list = []
    if os.path.exists(CORPUS_FILE):
        check_list.append("### 问答语料库文件检查完成 ###")
        if os.path.exists(MODEL_FILE):
            check_list.append("### 词向量模型文件检查完成 ###")
        else:
            check_list.append("### 未发现词向量模型文件，开始训练词向量 ###")
            corpus_training()
            check_list.append("### 词向量文件模型训练完毕 ###")

        if os.path.exists(MATRIX_FILE):
            check_list.append("### 问答矩阵文件检查完成 ###")
        else:
            check_list.append("### 未发现问答矩阵文件，开始计算问答矩阵 ###")
            model = word2vec.Word2Vec.load(MODEL_FILE)
            title, question, answer = TQR_import()
            q_matrix, fdist, singular_v = question_matrix(model, title, 'SIF')
            QD = QuestionDatabase(q_matrix, fdist, singular_v)
            with open(MATRIX_FILE, 'wb') as f:
                pickle.dump(QD, f, 0)
            check_list.append("### 问答矩阵文件计算完成 ###")
        if os.path.exists(MODEL_FILE) and os.path.exists(MATRIX_FILE):
            check_list.append("### 检索式回答机制文件检查完成 ###")
            check_flag = True
        else:
            check_list.append("### 检索式回答机制文件有误，请重新检查 ###")
            check_flag = False
    else:
        check_list.append("### 未发现问答语料库文件，进程终止，请检查 ###")
        check_flag = False
    # jieba_initial = jieba.cut("初始化结巴~~")
    if print_flag:
        print('\n'.join(check_list))
        print("——————————————————————————")

    return check_flag


# 加载【检索式方法】所需的数据
def Searching_load():
    # 训练word2vec词向量
    if not os.path.exists(MODEL_FILE):
        corpus_training()
    # 加载词向量模型
    model = word2vec.Word2Vec.load(MODEL_FILE)

    # 加载问答数据
    title, question, answer = TQR_import()

    # 构建问句SIF向量矩阵
    if not os.path.exists(MATRIX_FILE):
        q_matrix, fdist, singular_v = question_matrix(model, title, 'SIF')
        QD = QuestionDatabase(q_matrix, fdist, singular_v)
        with open(MATRIX_FILE, 'wb') as f:
            pickle.dump(QD, f, 0)
    else:
        q_matrix, fdist, singular_v = load_QD(MATRIX_FILE)

    return model, title, question, answer, q_matrix, fdist, singular_v


"""以下文件仅用于测试过程，开发完成后可弃用"""


# 原始回答文件的初步整体处理
def raw_reply_processing(l_min=4, l_max=100):
    """
    :return: 回答id，回答内容
    """
    raw_r = read_file("data/LSH_reply.txt", 0, 0)  # count 44729+1
    index_r_in = []
    r0_in = []
    for line in raw_r[1:]:
        try:
            r = line.replace('"', '').strip().split('\t')
            if l_min <= len(r[2]) <= l_max:
                index_r_in.append(int(r[1]))
                r0_in.append(zhengze(r[2].strip()))
        except IndexError:
            pass
    return index_r_in, r0_in


# 原始问句文件的初步整体处理，只保留有回答的问题
def raw_que_processing(l_min=5, l_max=100, del_flag=True):
    """
    :return: 问题id，问题主题，问题内容
    """
    raw_q = read_file("data/LSH_questions.txt", 0, 0)  # count 24294+1
    q0_in = []  # 问题主题
    q1_in = []  # 问题详细内容
    index_q_in = []
    if del_flag:  # 删除无回答的问句
        index_r_in, _ = raw_reply_processing()
        r_id_in = set(index_r_in)
        for line in raw_q[1:]:
            try:
                q = line.replace('"', '').strip().split('\t')
                if l_min <= len(q[1]) <= l_max:
                    q_id_in = int(q[0])
                    if q_id_in in r_id_in:
                        index_q_in.append(q_id_in)
                        q0_in.append(' '.join(jieba.cut(zhengze(q[1].strip()))))
                        q1_in.append(' '.join(jieba.cut(zhengze(q[2].strip()))))
            except IndexError:
                pass
    else:  # 保留所有问句
        for line in raw_q[1:]:
            try:
                q = line.replace('"', '').strip().split('\t')
                if l_min <= len(q[1]) <= l_max:
                    index_q_in.append(int(q[0]))
                    q0_in.append(' '.join(jieba.cut(zhengze(q[1].strip()))))
                    q1_in.append(' '.join(jieba.cut(zhengze(q[2].strip()))))
            except IndexError:
                pass
    return index_q_in, q0_in, q1_in


# 旧有构建专用语料库
def corpus_processing():
    """

    :return:
    """
    r_id_in, r0_in = raw_reply_processing(0)
    q_id_in, q0_in, q1_in = raw_que_processing(0)
    qq = []
    rr = []
    for i in range(len(q_id_in)):
        qq.append(' '.join(jieba.cut(q0_in[i] + q1_in[i])))
    for j in range(len(r_id_in)):
        rr.append(' '.join(jieba.cut(r0_in[j])))
    with open("data/LSH_QR_fenci.txt", 'a', encoding='utf-8') as f:
        for a in qq + rr:
            f.write(a + '\n')

    return qq + rr


# 测试question_base.py
def test1():
    q_id, q0, q1 = raw_que_processing()
    model = word2vec.Word2Vec.load("data/my_model.model")
    print("load model successfully")
    q0 = np.array(q0)
    print(q0.shape)
    q_matrix, fdist, singular_v = question_matrix(model, q0, 'SIF')
    print("generate question matrix successfully")
    print(q0[23])
    print(q_matrix[23])
    # QD = QuestionDatabase(q_matrix, fdist, singular_v)
    # with open('data/QD.txt', 'wb') as f:
    #     pickle.dump(QD, f, 0)
    # print("question database saved successfully")
    return 0


# 测试answer_generation.py
def test2():
    q_id, q0, q1 = raw_que_processing()
    model = word2vec.Word2Vec.load("data/my_model.model")
    print("load model successfully")
    q_matrix, fdist, singular_v = load_QD("data/QD.txt")
    print("generate question matrix successfully")
    while True:
        # id_test = int(input("ID:"))
        # print(q0[id_test].replace(' ', ''))
        q_input = input(">:")
        q_vec = que2vec(q_input, model, fdist, singular_v, 'AVG')
        res = top5_question(q_vec, q_matrix)
        print(res)
        print(q0[res[0]])


# reshape向量形状，改好
def test3():
    # q_id, q0, q1 = raw_que_processing()

    model = word2vec.Word2Vec.load("data/my_model.model")
    q_matrix, fdist, singular_v = load_QD("data/QD.txt")
    sim_dict = {}
    print(q_matrix[2])
    print(q_matrix[2].shape)
    # QM = q_matrix.reshape(1, 12566)
    # print(QM.shape)
    q_input = "咨询心里问题"
    q_vec = que2vec(q_input, model, fdist, singular_v, 'AVG')
    q_vec = q_vec.reshape(1, 100)
    print(q_vec.shape)
    # print(cosine_similarity(q_vec, q_matrix[2].reshape(1, 100)))
    print(q_matrix.shape[0])
    for i in range(12566):
        try:
            sim = cosine_similarity(q_vec, q_matrix[i].reshape(1, 100))[0][0]
            sim_dict[i] = sim
        except ValueError:
            # print(i)
            # print(q_matrix[i])
            # print(q0[i])
            pass

    d = sorted(sim_dict, key=lambda x: sim_dict[x], reverse=True)
    print(d[:5])
    # print(sim_dict)
    # a = np.array([[3, 4, 5, 6, 7]])
    # print(a.shape)
    # b = np.array([[4, 5, 7, 1, 2], [3, 4, 5, 1, 3]])
    # print(b.shape)
    # print(cosine_similarity(a, b))
    # sim = cosine_similarity(q_vec, q_matrix[1])
    # print(sim)


# 检查sif问题，重新更改矩阵形式
def test4():
    q_id, q0, q1 = raw_que_processing()

    # model = word2vec.Word2Vec.load("data/my_model.model")
    # print("load model successfully")
    # vec_test = np.array(s2v(model, "加油加油"))
    # print(vec_test.shape)
    # print(vec_test)
    # corpus_test = ['到底 该不该 辞职', '感觉 自己 人生 快要 毁 了', '他 是 在 拒绝 我 联系 他 吗 ？']
    # corpus_test = q0[:100]
    # print(corpus_test)
    #
    # q_matrix1, _, _ = question_matrix(model, corpus_test, 'SIF')
    # #
    # print(q_matrix1.shape)
    ll = []
    for i in q0:
        ll.append(len(i))
    ll = np.sort(np.array(ll))
    print(ll[:10])


# 第一遍跑完后重新走一遍试一下
def test5():
    # 训练word2vec词向量
    if not os.path.exists(MODEL_FILE):
        corpus_training()
        print("model trained successfully")

    # 加载词向量模型
    model = word2vec.Word2Vec.load(MODEL_FILE)
    print("model loaded successfully")

    # 加载问答数据
    title, question, answer = TQR_import()
    print("TQR_data loaded successfully")

    # 构建问句SIF向量矩阵
    if not os.path.exists(MATRIX_FILE):
        q_matrix, fdist, singular_v = question_matrix(model, title, 'SIF')
        QD = QuestionDatabase(q_matrix, fdist, singular_v)
        print("question database calculated successfully ")
        with open(MATRIX_FILE, 'wb') as f:
            pickle.dump(QD, f, 0)
        print("question database saved successfully")
    else:
        q_matrix, fdist, singular_v = load_QD(MATRIX_FILE)
        print("Q_vec_matrix loaded successfully ")

    # print("问题模型加载完毕")
    # print(q_matrix.shape)
    # for line in title:
    #     # print(line)
    #     print(s2v(model, line).shape)
    # print(q_matrix.shape)
    # print(fdist)

    while True:
        q_in = input('>:')
        q_vec = que2vec(q_in, model, fdist, singular_v, 'SIF')
        res = top5_question(q_vec, q_matrix)
        # print(res)
        # print("置信度：", res[0][1])
        print("-:", answer[res[0][0]].replace(" ", ""))


# 提前初始化结巴
def initial_jieba(sentence_in):
    return ''.join(jieba.cut(sentence_in))


if __name__ == '__main__':
    pass

