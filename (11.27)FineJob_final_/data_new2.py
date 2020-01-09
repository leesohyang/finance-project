from collections import defaultdict
from tqdm import tqdm_notebook
import random
from collections import Counter
from konlpy.tag import Okt
import pandas as pd
from eunjeon import Mecab
# from tqdm import tqdm_notebook
import numpy as np


def save_vocab(filename, vocab):
    with open(filename, 'w', encoding='utf-8') as f:
        for v in vocab:
            f.write('%s\t%d\n' % (v, vocab[v]))


def load_vocab(filename):
    result = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ls = line.split('\t')
            result[ls[0]] = int(ls[1])

    return result


def data_loader_for_test(input): #얘 수정하기.
    tag_count = []
    mecab = Okt()
    Mecab_tag = mecab.pos(input)
    tag_count.append(Counter(Mecab_tag).most_common())
    mecab.morphs(input)

    go_pos = ['Noun', 'Adjective', 'Josa']

    word = []
    for tag in Mecab_tag:
        if tag[1] in go_pos:
            # if tag[0] not in stopWord:
            if len(tag[0]) > 1:
                word.append(tag[0])
    return word


def data_loader_last_for_test(total_word, vocab):  # 한 줄이 입력으로 들어옴.
    result = []
    sequence = []
    # for i, sentence in enumerate(total_word): totalword=2차원배열
    #     sequence = [get_token_id(t, vocab) for t in sentence]
    for token in total_word:
        sequence.append(get_token_id(token, vocab))

    seq_seg = sequence[:50]
    seq_len = []
    seq_len.append(len(seq_seg))

    padding = [1] * (50 - len(seq_seg))
    seq_seg = seq_seg + padding

    result.append((seq_seg, seq_len))

    return result

    # while len(sequence) > 0:
    #     seq_seg = sequence[:50]
    #     seq_len = []
    #     seq_len.append(len(seq_seg))
    #     sequence = sequence[50:]
    #
    #     padding = [1] * (50 - len(seq_seg))
    #     seq_seg = seq_seg + padding
    #
    #     result.append(((seq_seg, seq_len)))

    # return result


def data_loader_raw():
    total_data = pd.read_csv(r"C:\Users\이소향\Downloads\abc.csv")
    total_data['장점'] = total_data['장점'].str.replace("[^a-zA-Zㄱ-ㅎ가-힣0-9]", " ")

    # total_data = total_data[:100]
    total_word = []
    tag_count = []

    for i in range(len(total_data)):

        mecab = Okt()
        mecab_text = total_data['장점'][i]
        Mecab_tag = mecab.pos(mecab_text)

        tag_count.append(Counter(Mecab_tag).most_common())
        mecab.morphs(mecab_text)

        go_pos = ['Noun', 'Adjective', 'Josa']

        word = []
        for tag in Mecab_tag:
            if tag[1] in go_pos:
                # if tag[0] not in stopWord:
                if len(tag[0]) > 1:
                    word.append(tag[0])

        total_word.append(word)

    total_label = total_data['label'][:].values.tolist()

    # token2idx = defaultdict(lambda: len(token2idx))
    # pad = token2idx['<PAD>']
    print("building vocab")
    vocab = build_vocab(total_word)
    print(len(vocab))

    # 패딩 전 input
    # input_x = list(get_token_id(total_word, vocab))
    # x_test = list(convert_token_to_idx(x_test))

    return total_word, total_label, vocab


def build_vocab(token_ls):
    vocab = dict()
    vocab['#UNKOWN'] = 0
    vocab['#PAD'] = 1
    for tokens in token_ls:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def get_onehot(index, size):
    onehot = [0] * size
    onehot[index] = 1
    return onehot


def get_token_id(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        0


def data_loader_last(total_word, total_label, vocab):
    result = []

    for i, sentence in enumerate(total_word):
        sequence = [get_token_id(t, vocab) for t in sentence]

        #     seq_seg = sequence[:50]
        #     seq_len = []
        #     seq_len.append(len(seq_seg))

        #     padding = [1] * (50 - len(seq_seg))
        #     seq_seg = seq_seg + padding

        #     result.append((seq_seg, get_onehot(total_label[i], 3), seq_len))

        # return result

        while len(sequence) > 0:
            seq_seg = sequence[:50]
            seq_len = []
            seq_len.append(len(seq_seg))
            sequence = sequence[50:]

            padding = [1] * (50 - len(seq_seg))
            seq_seg = seq_seg + padding

            result.append(((seq_seg, get_onehot(total_label[i], 3), seq_len)))

    return result

# ---데이터와 라벨의 dataframes-----
# input_data = total_data['장점']
# input_label = total_data['label']
# ---------------------------------