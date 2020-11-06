# 自动聊天机器人

《面向对象Java》课程大作业，基于检索式+生成式构建的心理健康主题的自动聊天机器人。The Java course homework, an automatic chat robot based on the mental health theme constructed by retrieval & generative.

原始语料库来自其他小组的爬取的心理健康主题问答对，质量参差不齐。

#### 主要实现思路：

- 利用GloVe工具实现词嵌入，SIF法构建驹向量；
- 对于有直接答案的简单问题，直接通过余弦相似度搜索一致的现成答案；
- 训练了一个seq2seq的循环神经网络，在解码方向加入attention机制，对于相似度低于阈值（80%）的复杂问题，使用网络生成答案。

#### 使用方法：

- 下载语料库文件，将该`data`文件夹直接放入项目根目录，地址：[Google Drive](https://drive.google.com/drive/folders/1JeFioglqzN3Den4VanfvqPsdsyTDO77k?usp=sharing)，

- `pip install -r requirements.txt `，或者手动安装以下包

  ```bash
  xlwings==0.19.5
  jieba==0.42.1
  torch==1.6.0
  gensim==3.8.3
  nltk==3.5
  scikit-learn==0.23.1
  numpy==1.18.5
  ```

- 直接运行`Demo.py`即可。

详细的技术文档见`report.pdf`