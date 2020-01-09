import pandas as pd
import os
import numpy as np

def data_loader_X(path):
    # fd = pd.read_excel('sample.xlsx')
    # path = 'C:/20190530_final/KISVALUE_상폐/'
    # path = 'C:/20190530_final/data_sample/'
    files = os.listdir(path)
    col = ["Date", "0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
           "0C1103/PBR"]
    result = []
    for file in files:

        n_path = path + file

        df = pd.read_excel(n_path)

        fd = df[5:]
        if len(fd.columns) == 15:
            break
        fd.columns = col  # column 새로 지정
        # # Date열을 제외한 모든 행의 값이 NaN인 경우 행 삭제
        fd = fd.dropna(
            subset=["0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
                    "0C1103/PBR"],
            how='all')

        # print(fd.drop(labels = ['Unnamed: 0', 'Date'], axis=1))
        fd2 = fd.drop(labels=['Date'], axis=1)

        fd2['label'] = 0
        fd2['len'] = len(fd2.index)
        fd2 = fd2.reset_index(drop=True)
        a = len(fd2.index)
        # print(a)
        if a < 274:
            b = 275 - a
            for i in range(b):
                fd2.loc[a + i] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # print(len(fd2.index))
        fd2 = fd2.fillna(0)
        fd_f = fd2.values

        input_data = fd_f[:, :-2]
        input_data_norm = min_max_scaling(input_data)
        # label = fd_f[1, -2]
        label = [1,0]
        len_stock = fd_f[1, -1]
        # print(type(len_stock))
        # label = label.tolist()
        input_data = input_data.tolist()
        # len_stock = len_stock.tolist()
        result.append((input_data_norm, label, [len_stock]))
        # print(result)

    return result

def data_loader_O(path):
    # fd = pd.read_excel('sample.xlsx')
    # path = 'C:/20190530_final/KISVALUE_계속/'
    files = os.listdir(path)
    col = ["Date", "0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
           "0C1103/PBR"]
    result = []
    for file in files:

        n_path = path + file

        df = pd.read_excel(n_path)
       
        fd = df[5:]
        fd.columns = col  # column 새로 지정
        # # Date열을 제외한 모든 행의 값이 NaN인 경우 행 삭제
        fd = fd.dropna(
            subset=["0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
                    "0C1103/PBR"],
            how='all')

        # print(fd.drop(labels = ['Unnamed: 0', 'Date'], axis=1))
        fd2 = fd.drop(labels=['Date'], axis=1)

        fd2['label'] = 0
        fd2['len'] = len(fd2.index)
        fd2 = fd2.reset_index(drop=True)
        a = len(fd2.index)
        # print(a)
        if a < 274:
            b = 275 - a
            for i in range(b):
                fd2.loc[a + i] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # print(len(fd2.index))
        fd2 = fd2.fillna(0)
        fd_f = fd2.values
        input_data = fd_f[:, :-2]
        input_data_norm = min_max_scaling(input_data)
        # label = fd_f[1, -2]
        label = [0,1]
        len_stock = fd_f[1, -1]
        # print(type(len_stock))
        # label = label.tolist()
        input_data = input_data.tolist()
        # len_stock = len_stock.tolist()
        result.append((input_data_norm, label, [len_stock]))
        # print(result)

    return result


def data_loader_O_for_test(company_code):
    # fd = pd.read_excel('sample.xlsx')
    path = 'C:/20190530_final/data_sample/'
    path = path + company_code + '.xls'
    # files = os.listdir(path)
    col = ["Date", "0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
           "0C1103/PBR"]
    result = []
    # for file in files:

        # n_path = path + file

    df = pd.read_excel(path)

    fd = df[5:]
    fd.columns = col  # column 새로 지정
    # # Date열을 제외한 모든 행의 값이 NaN인 경우 행 삭제
    fd = fd.dropna(
        subset=["0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
                "0C1103/PBR"],
        how='all')

    # print(fd.drop(labels = ['Unnamed: 0', 'Date'], axis=1))
    fd2 = fd.drop(labels=['Date'], axis=1)

    fd2['label'] = 0
    fd2['len'] = len(fd2.index)
    fd2 = fd2.reset_index(drop=True)
    a = len(fd2.index)
    # print(a)
    if a < 274:
        b = 275 - a
        for i in range(b):
            fd2.loc[a + i] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # print(len(fd2.index))
    fd2 = fd2.fillna(0)
    fd_f = fd2.values
    input_data = fd_f[:, :-2]
    input_data_norm = min_max_scaling(input_data)
    # label = fd_f[1, -2]
    label = [0, 1]
    len_stock = fd_f[1, -1]
    # print(type(len_stock))
    # label = label.tolist()
    input_data = input_data.tolist()
    # len_stock = len_stock.tolist()
    result.append((input_data_norm, label, [len_stock]))
    # print(result)

    return result

def data_standardization(x):
    x_np = np.asarray
    return (x_np - x.np.mean()) / x_np.std()

def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)

# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 Org_X값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return(x_np + (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

# --------------위에는 build
def save_data(filename, data):
    def make_csv_str(d):
        output = '%d' % d[0]
        for index in d[1:]:
            output = '%s,%d' % (output, index)
        return output

    with open(filename, 'w', encoding='utf-8') as f:
        for d in data:
            data_str = make_csv_str(d[0])
            label_str = make_csv_str(d[1])
            lens_str = make_csv_str(d[2])
            f.write (data_str + '\n')
            f.write (label_str + '\n')
            f.write (lens_str + '\n')

def load_data(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(int(len(lines)/3)):
            data = lines[i*3] #각각 리스트임. 일차원
            label = lines[i*3 + 1]
            lens = lines[i*3 + 2]
            result.append((([int(s) for s in data.split(',')], [int(s) for s in label.split(',')], [int(lens)])))
    return result


# ------shuffle
# shuffle_indices = np.random.permutation(np.arange(data_size))
#
# data = data[shuffle_indices]
#
# labels = labels[shuffle_indices]
#
# lengths = lengths[shuffle_indices]