# -*- coding: utf-8 -*-
# @Time    : 20/02/09 8:31
# @Author  : Wang Yu
# @Project : CHAT
# @File    : Generating.py
# @Software: PyCharm


"""生成式回答机制"""


import jieba
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

PAD_token, SOS_token, EOS_token = 0, 1, 2
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
NN_MODEL_FILE = "data/NN-model.tar"


# 检查各种文件是否存在
def Generating_check(print_flag):
    if os.path.exists(NN_MODEL_FILE):
        if print_flag:
            print("### 生成式问答机制文件检查完成  ###")
            print("——————————————————————————")

        return True
    else:
        if print_flag:
            print("### 生成式问答机制文件有误，请重新检查  ###")
        return False


# 加载训练好的模型数据
def Generating_load():
    corpus_name = "dialog1000k"
    """调参"""
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 4
    dropout = 0.1

    # 设置加载的检查点，若从头开始则设定为无
    # load_filename = None
    # load_filename = os.path.join(save_dir, model_name, corpus_name,
    #                              f"{encoder_n_layers}-{decoder_n_layers}_{hidden_size}",
    #                              f"{checkpoint_iter}_checkpoint.tar")

    # 定义Voc类
    class Voc:
        def __init__(self, name):  # 类中加入的name属性从头到尾就没用过
            self.name = name
            self.trimmed = False
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count SOS, EOS, PAD

        def add_sentence(self, sentence):
            for word in sentence.split(' '):
                self.add_word(word)

        def add_word(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                self.word2count[word] += 1

        # 删除低于特定计数阈值的单词， 修剪函数
        def trim(self, min_count):
            if self.trimmed:
                return
            self.trimmed = True
            keep_words = []
            for k1, v1 in self.word2count.items():
                if v1 >= min_count:
                    keep_words.append(k1)
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3
            for word in keep_words:
                self.add_word(word)

    class EncoderRNN(nn.Module):
        def __init__(self, hidden_size_in, embedding_in, n_layers=1, dropout_in=0):
            super(EncoderRNN, self).__init__()
            self.n_layers = n_layers
            self.hidden_size = hidden_size_in
            self.embedding = embedding_in

            # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
            # because our input size is a word embedding with number of features == hidden_size
            self.gru = nn.GRU(hidden_size_in, hidden_size_in, n_layers,
                              dropout=(0 if n_layers == 1 else dropout_in), bidirectional=True)

        def forward(self, input_seq, input_lengths, hidden_in=None):
            # Convert word indexes to embeddings
            embedded = self.embedding(input_seq)
            # Pack padded batch of sequences for RNN module
            packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
            # Forward pass through GRU
            outputs_out, hidden_out = self.gru(packed, hidden_in)
            # Unpack padding 解包
            outputs_out, _ = nn.utils.rnn.pad_packed_sequence(outputs_out)
            # Sum bidirectional GRU outputs
            outputs_out = outputs_out[:, :, :self.hidden_size] + outputs_out[:, :, self.hidden_size:]
            # Return output and final hidden state
            return outputs_out, hidden_out

    # Luong attention layer
    class Attn(nn.Module):
        def __init__(self, method, hidden_size_in):
            super(Attn, self).__init__()
            self.method = method
            if self.method not in ['dot', 'general', 'concat']:
                raise ValueError(self.method, "is not an appropriate attention method.")
            self.hidden_size = hidden_size_in
            if self.method == 'general':
                self.attn = nn.Linear(self.hidden_size, hidden_size_in)
            elif self.method == 'concat':
                self.attn = nn.Linear(self.hidden_size * 2, hidden_size_in)
                self.v = nn.Parameter(torch.FloatTensor(hidden_size_in))

        @staticmethod
        def dot_score(hidden, encoder_output):
            return torch.sum(hidden * encoder_output, dim=2)

        def general_score(self, hidden, encoder_output):
            energy = self.attn(encoder_output)
            return torch.sum(hidden * energy, dim=2)

        def concat_score(self, hidden, encoder_output):
            energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
            return torch.sum(self.v * energy, dim=2)

        def forward(self, hidden, encoder_outputs):
            # Calculate the attention weights (energies) based on the given method
            attn_energies = None
            if self.method == 'general':
                attn_energies = self.general_score(hidden, encoder_outputs)
            elif self.method == 'concat':
                attn_energies = self.concat_score(hidden, encoder_outputs)
            elif self.method == 'dot':
                attn_energies = self.dot_score(hidden, encoder_outputs)

            # Transpose max_length and batch_size dimensions
            attn_energies = attn_energies.t()

            # Return the softmax normalized probability scores (with added dimension)
            return F.softmax(attn_energies, dim=1).unsqueeze(1)

    class LuongAttnDecoderRNN(nn.Module):
        def __init__(self, attn_model_in, embedding_in, hidden_size_in, output_size_in, n_layers_in=1, dropout_in=0.1):
            super(LuongAttnDecoderRNN, self).__init__()
            # Keep for reference
            self.attn_model = attn_model_in
            self.hidden_size = hidden_size_in
            self.output_size = output_size_in
            self.n_layers = n_layers_in
            self.dropout = dropout_in
            # 定义层
            self.embedding = embedding_in
            self.embedding_dropout = nn.Dropout(dropout_in)
            self.gru = nn.GRU(hidden_size_in, hidden_size_in, n_layers_in,
                              dropout=(0 if n_layers_in == 1 else dropout_in))
            self.concat = nn.Linear(hidden_size_in * 2, hidden_size_in)
            self.out = nn.Linear(hidden_size_in, output_size_in)
            self.attn = Attn(attn_model_in, hidden_size_in)

        def forward(self, input_step_in, last_hidden_in, encoder_outputs_in):
            # 注意：一次运行此步骤（单词）
            # 获取当前输入词的嵌入
            embedded = self.embedding(input_step_in)
            embedded = self.embedding_dropout(embedded)
            # 通过单向GRU转发
            rnn_output, hidden_out = self.gru(embedded, last_hidden_in)
            # 根据当前GRU输出计算注意力权重
            attn_weights = self.attn(rnn_output, encoder_outputs_in)
            # 将注意力权重乘以编码器输出以获得新的“加权总和“上下文向量
            context = attn_weights.bmm(encoder_outputs_in.transpose(0, 1))
            # 使用Luong连接甲醛上下文向量和GRU输出
            rnn_output = rnn_output.squeeze(0)
            context = context.squeeze(1)
            concat_input = torch.cat((rnn_output, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            # 使用Luong预测下一个单词
            output_out = self.out(concat_output)
            output_out = F.softmax(output_out, dim=1)
            # 返回输出并最终隐藏状态
            return output_out, hidden_out

    class GreedySearchDecoder(nn.Module):
        def __init__(self, encoder_in, decoder_in):
            super(GreedySearchDecoder, self).__init__()
            self.encoder = encoder_in
            self.decoder = decoder_in

        def forward(self, input_seq, input_length, max_length):
            # 通过解码器模型的正向输入
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
            # 准备编码器的最终隐藏层，使其成为解码器的第一个隐藏输入
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            # 使用SOS_token初始化解码器输入
            decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
            # 初始化张量以附加解码后的单词
            all_tokens = torch.zeros([0], device=device, dtype=torch.long)
            all_scores = torch.zeros([0], device=device)
            # 每次迭代解码一个token
            for _ in range(max_length):
                # 前向通过解码器
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # 获得最可能的单词标记及其softmax分数
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                # 记录token和分数
                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                all_scores = torch.cat((all_scores, decoder_scores), dim=0)
                # 准备当前token作为下一个解码器输入（添加维度）
                decoder_input = torch.unsqueeze(decoder_input, 0)
            # 返回单词标记和分数的集合
            return all_tokens, all_scores

    voc = Voc(corpus_name)

    if os.path.exists(NN_MODEL_FILE):
        checkpoint = torch.load(NN_MODEL_FILE)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
        embedding = nn.Embedding(voc.num_words, hidden_size)
        embedding.load_state_dict(embedding_sd)
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        encoder.eval()
        decoder.eval()
        searcher = GreedySearchDecoder(encoder, decoder)
        return searcher, voc
    else:
        print("seq2seq模型文件不存在！")
        return None


def indexes_from_sentence(voc_in, sentence_in):
    return [voc_in.word2index[word] for word in sentence_in.split(' ')] + [EOS_token]


def evaluate(searcher_in, voc_in, sentence_in, max_length=30):
    # 批量格式化输入句子
    # 单词->索引
    indexes_batch = [indexes_from_sentence(voc_in, sentence_in)]
    # 创建长度张量
    lengths_in = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转换批次的尺寸以符合模型的期望
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 转换设别
    input_batch = input_batch.to(device)
    lengths_in = lengths_in.to(device)
    # 搜索器解码句子
    tokens, scores = searcher_in(input_batch, lengths_in, max_length)
    # 索引->单词
    decoded_words = [voc_in.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluate_input(searcher_in, voc_in, input_sentence):
    try:
        input_sentence = ' '.join(jieba.cut(input_sentence.strip()))
        # 句子评价
        output_words = evaluate(searcher_in, voc_in, input_sentence)
        # 格式化及打印输出
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        return ''.join(output_words)

    except KeyError:
        # print("Error: Encountered unknown word.")
        return "对不起我没法帮助您，请咨询专业的心理医生"


if __name__ == '__main__':
    pass
