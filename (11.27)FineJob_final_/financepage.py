import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
import random
from bs4 import BeautifulSoup

matplotlib.rcParams['font.family'] = "AppleGothic"
matplotlib.rcParams['figure.figsize'] = (35, 20)
matplotlib.rcParams['font.size'] = 20

# 크롤링해올 주소를 만들기 위한 dataIdx와 회사 상호명 Dictionary

# dataIdx_dic = {"알테오젠": "196170", "제넨바이오": "072520", "케이피티유": "054410"}
dataIdx_dic = {"이녹스(주)": "088390"}

def getName(review):
    name = '페이스북코리아(유)' # company페이지에서 선택한 기업명
    return name

def getInterview(name):
    df_m = pd.read_csv('static/interview.csv', index_col='기업')
    ls = list(df_m.loc[name, '면접 질문'].values)
    ls = random.sample(ls, 2)
    return ls[0], ls[1]

def getReview(name):
    df_r = pd.read_csv('static/장점.csv', index_col='기업')
    ls_r = list(df_r.loc[name, '장점'].values)
    return ls_r

def makeCompanybodyLink(company_code): #비즈니스 서머리
    search_link_format = "http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A"
    search_link_format2 = "&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701"
    try:
        # inputString = inputString.upper()
        # dataIdx = dataIdx_dic[inputString]
        # encodingName = requests.utils.quote(inputString)
        link = search_link_format + company_code + search_link_format2
    except:
        print("{}는 아직 서비스를 지원하지 않습니다.".format(company_code))
        link = "error"

    return link

def makeCompanyInfoLink2(company_code): #회사정보1, 2, 매출비중추이
    search_link_format = "http://comp.fnguide.com/SVO2/ASP/SVD_Corp.asp?pGB=1&gicode=A"
    search_link_format2 = "&cID=&MenuYn=Y&ReportGB=&NewMenuID=102&stkGb=701"
    try:
        # inputString = inputString.upper()
        # dataIdx = dataIdx_dic[inputString]
        # encodingName = requests.utils.quote(inputString)
        link2 = search_link_format + company_code + search_link_format2
    except:
        print("{}는 아직 서비스를 지원하지 않습니다.".format(company_code))
        link = "error"

    return link2

def makeCompanyInfoLink3(company_code): #경쟁사비교표
    search_link_format = "http://comp.fnguide.com/SVO2/ASP/SVD_Comparison.asp?pGB=1&gicode=A"
    search_link_format2 = "&cID=&MenuYn=Y&ReportGB=&NewMenuID=102&stkGb=701"
    try:
        # inputString = inputString.upper()
        # dataIdx = dataIdx_dic[inputString]
        # encodingName = requests.utils.quote(inputString)
        link3 = search_link_format + company_code + search_link_format2
    except:
        print("{}는 아직 서비스를 지원하지 않습니다.".format(company_code))
        link = "error"

    return link3

def makeBusinessSummary(company):

    company_code = dataIdx_dic[company]
    link = makeCompanybodyLink(company_code)
    req = requests.get(link)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

    # Business Summary 가져오기(제목, 바디)
    bsum_title = soup.select('#bizSummaryHeader')[0].text
    bsum_body = soup.select('#bizSummaryContent')[0].text
    return bsum_title, bsum_body

def makesalegraph(company_code):
    link2 = makeCompanyInfoLink2(company_code)
    req = requests.get(link2)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    img = soup.find("img")
    img_src = img.get("src")
    sales_graph = "http://comp.fnguide.com/" + img_src
    return sales_graph

# 이 밑으로 데이터프레임
def makeCompanyInfo1(company_code):
    link2 = makeCompanyInfoLink2(company_code)
    dataframes_comp = pd.read_html(link2)
    df = dataframes_comp[0]
    df.columns = ['', '', '', '']
    df.index = ['', '', '', '', '', '', '', '', '', '']
    dataframes_comp = df
    com_info001 = dataframes_comp.iloc[1:10, 0:2].fillna('-')
    return com_info001
    # com_info001 = dataframes_comp[0].iloc[1:10, 0:2].fillna('-')
    # # com_info002 = dataframes_comp[0].iloc[1:10, 2:4]
    # # print(com_info001)
    # com_info001.index = com_info001[0]
    # com_info01 = com_info001.iloc[0:10, 1:2]
    # return com_info01

def makeCompanyInfo2(company_code):
    link2 = makeCompanyInfoLink2(company_code)
    dataframes_comp = pd.read_html(link2)
    df = dataframes_comp[0]
    df.columns = ['', '', '', '']
    df.index = ['', '', '', '', '', '', '', '', '', '']
    dataframes_comp = df
    com_info002 = dataframes_comp.iloc[1:10, 2:4].fillna('-')
    return com_info002

def makeCompetInfo(company_code):
    link3 = makeCompanyInfoLink3(company_code)
    dataframes_comp = pd.read_html(link3)
    comparision_data = dataframes_comp[0].fillna('-')
    comparision_table = comparision_data.set_index('구분')
    return comparision_table

def makeDataframeFromLink(company_code):

    # 가져올거 네개 세개는 데이터프레임 한개는 그래프
    # make_df01 = makeBusinessSummary(company_code) 얘는 string임.
    make_df02 = makeCompanyInfo1(company_code)
    make_df03 = makeCompanyInfo2(company_code)
    make_df04 = makeCompetInfo(company_code)
    # df04는 엑셀파일로 긁어와야지. app에서 했음
    # make_df04 = makeCompanyFinance_stock()

    return [make_df02, make_df03, make_df04]
# def makePathFile(inputString):
#     dataCode = dataIdx_dic[inputString]
#     return dataCode

def searchAndMakeDataframe(company):
    # inputString = 종목코드
    company_code = dataIdx_dic[company]
    link = makeCompanybodyLink(company_code)
    link2 = makeCompanyInfoLink2(company_code)
    link3 = makeCompanyInfoLink3(company_code)
    if (link != 'error') & (link2 != 'error') & (link3 != 'error'):
        dataframes = makeDataframeFromLink(company_code)
    return dataframes

def makeCompanyFinance_stock(company):

    company_code = dataIdx_dic[company]
    path = 'C:/20190530_final/data_sample/088390.xls'
    col = ["Date", "0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
           "0C1103/PBR"]
    result = []
    df = pd.read_excel(path)

    fd = df[5:]

    fd.columns = col  # column 새로 지정
    # # Date열을 제외한 모든 행의 값이 NaN인 경우 행 삭제
    fd = fd.dropna(
        subset=["0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
                "0C1103/PBR"],
        how='all')
    make_df_stock = pd.DataFrame()
    make_df_stock['Stock'] = fd["0C1005/수정주가"]
    make_df_stock.index = fd["Date"]
    #지금 내가 그린건 막대그래프란다.
    # graph01 = make_df_stock.plot.bar(grid=True, width=0.7)
    graph01 = make_df_stock.plot.line(grid=True)
    graph01.set_title("수정주가 현황", size=20)
    graph01.set_xlabel("날짜", size=20)
    graph01.set_ylabel("주가", size=20)
    # graph01.set_ylim(10000, 20000)
    plt.xticks(rotation=5, size=17)
    # plt.yticks(size=17)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    chart = 'data:image/png;base64,{}'.format(graph_url)

    return chart
# Business Summary + graph
# def makeCompanybodyLink(inputString):
#     search_link_format = "http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A"
#     search_link_format2 = "&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701"
#     try:
#         inputString = inputString.upper()
#         dataIdx = dataIdx_dic[inputString]
#         encodingName = requests.utils.quote(inputString)
#         link = search_link_format + dataIdx + search_link_format2 + encodingName
#     except:
#         print("{}는 아직 서비스를 지원하지 않습니다.".format(inputString))
#         link = "error"
#
#     return link
#
# def makeCompanyInfoLink(inputString):
#     search_link_format = "http://comp.fnguide.com/SVO2/ASP/SVD_Corp.asp?pGB=1&gicode=A"
#     search_link_format2 = "&cID=&MenuYn=Y&ReportGB=&NewMenuID=102&stkGb=701"
#     try:
#         inputString = inputString.upper()
#         dataIdx = dataIdx_dic[inputString]
#         encodingName = requests.utils.quote(inputString)
#         link2 = search_link_format + dataIdx + search_link_format2 + encodingName
#     except:
#         print("{}는 아직 서비스를 지원하지 않습니다.".format(inputString))
#         link = "error"
#
#     return link2
# # Business Summary 가져오기
# def makeCompanyBusinessSummary(link):
#     # req = requests.get('http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A'+getCode('알테오젠')+'&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701')
#     req = requests.get(
#         'http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A' + '088390' + '&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701')
#     html = req.text
#     soup = BeautifulSoup(html, 'html.parser')
#
#     # Business Summary 가져오기(제목, 바디)
#     bsum_title = soup.select('#bizSummaryHeader')[0].text
#     bsum_body = soup.select('#bizSummaryContent')[0].text
#
#     return bsum_title, bsum_body
# # 회사개요 가져오기
# def makeCompanyInfoDataFrame(link):
#     dataframes_comp = pd.read_html(link)
#     # print(dataframes_comp[0])
#     com_info001 = dataframes_comp[0].iloc[1:10, 0:2].fillna('-')
#     # com_info002 = dataframes_comp[0].iloc[1:10, 2:4]
#     # print(com_info001)
#     com_info001.index = com_info001[0]
#     com_info01 = com_info001.iloc[0:10, 1:2]
#     return com_info01
#
# # 회사개요 가져오기2
# def makeCompanyInfoDataFrame_2(link):
#     dataframes_comp = pd.read_html(link)
#     com_info002 = dataframes_comp[0].iloc[1:10, 2:4].fillna('-')
#     com_info002.index = com_info002[2]
#     com_info02 = com_info002.iloc[0:10, 1:2]
#     return com_info02
#
# # 경쟁사비교표 가져오기
# def makeCompanycompareData(link):
#     dataframes_comp = pd.read_html(link)
#     comparision_data = dataframes_comp[0].fillna('-')
#     comparision_table = comparision_data.set_index('구분')
#     return comparision_table
#
# # 주가 그래프 그리기
# def makeCompanyFinance_stock(n_path):
#     col = ["Date", "0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
#            "0C1103/PBR"]
#     result = []
#     df = pd.read_excel(n_path)
#
#     fd = df[5:]
#
#     fd.columns = col  # column 새로 지정
#     # # Date열을 제외한 모든 행의 값이 NaN인 경우 행 삭제
#     fd = fd.dropna(
#         subset=["0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
#                 "0C1103/PBR"],
#         how='all')
#     make_df_stock = pd.DataFrame()
#     make_df_stock['Stock'] = fd["0C1005/수정주가"]
#     make_df_stock.index = fd["Date"]
#     #지금 내가 그린건 막대그래프란다.
#     # graph01 = make_df_stock.plot.bar(grid=True, width=0.7)
#     graph01 = make_df_stock.plot.line(grid=True)
#     graph01.set_title("수정주가 현황", size=20)
#     graph01.set_xlabel("날짜", size=20)
#     graph01.set_ylabel("주가", size=20)
#     graph01.set_ylim(10000, 20000)
#     plt.xticks(rotation=5, size=17)
#     # plt.yticks(size=17)
#
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     graph_url = base64.b64encode(img.getvalue()).decode()
#     plt.close()
#
#     chart = 'data:image/png;base64,{}'.format(graph_url)
#
#     return chart
#
# def makeDataframeFromLink(link, link2):
#
#     # 가져올거 네개 세개는 데이터프레임 한개는 그래프
#     make_df01 = makeCompanyInfoDataFrame(link)
#     make_df02 = makeCompanyInfoDataFrame_2(link)
#     make_df03 = makeCompanycompareData(link2)
#
#     # df04는 엑셀파일로 긁어와야지. app에서 했음
#     # make_df04 = makeCompanyFinance_stock()
#
#     return [make_df01, make_df02, make_df03]
#
# def searchAndMakeDataframe2(inputString):
#     # inputString = 종목코드
#     link = makeCompanyInfoLink(inputString)
#     link2 = makeCompanybodyLink(inputString)
#     if (link != 'error') & (link2 != 'error'):
#         dataframes = makeDataframeFromLink(link, link2)
#     return dataframes
#
# def searchAndMakeDataframe():
#     inputString = input() #inputString = 종목코드
#     link = makeCompanyInfoLink(inputString)
#     link2 = makeCompanybodyLink(inputString)
#     if (link != 'error') & (link2 != 'error'):
#         dataframes = makeDataframeFromLink(link, link2)
#     return dataframes
#
# def MultisearchAndMakeDataframe():
#     inputString = input()
#     inputString = inputString.split()
#     link01 = makeCompanyInfoLink(inputString[0])
#     link02 = makeCompanyInfoLink(inputString[1])
#     if link01 != 'error' and link02 != 'error':
#         dataframes01 = makeDataframeFromLink(link01)
#         dataframes02 = makeDataframeFromLink(link02)
#     else:
#         dataframes01 = ["Service is not offer"] * 8
#         dataframes02 = ["Service is not offer"] * 8
#     return [dataframes01, dataframes02]
#
# def MultisearchAndMakeDataframe2(inputString):
#     inputString = inputString.split()
#     link01 = makeCompanyInfoLink(inputString[0])
#     link02 = makeCompanyInfoLink(inputString[1])
#     if link01 != 'error' and link02 != 'error':
#         dataframes01 = makeDataframeFromLink(link01)
#         dataframes02 = makeDataframeFromLink(link02)
#     else:
#         dataframes01 = ["Service is not offer"] * 8
#         dataframes02 = ["Service is not offer"] * 8
#     return [dataframes01, dataframes02]
#
# def data_standardization(x):
#     x_np = np.asarray
#     return (x_np - x.np.mean()) / x_np.std()
#
# def min_max_scaling(x):
#     x_np = np.asarray(x)
#     return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)
#
# # 정규화된 값을 원래의 값으로 되돌린다
# # 정규화하기 이전의 Org_X값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴
# def reverse_min_max_scaling(org_x, x):
#     org_x_np = np.asarray(org_x)
#     x_np = np.asarray(x)
#     return(x_np + (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()
#
# def data_loader_X(path_file):
#     # fd = pd.read_excel('sample.xlsx')
#     path = 'C:/20190530/data/'
#     col = ["Date", "0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
#            "0C1103/PBR"]
#     result = []
#
#     df = pd.read_excel(path_file)
#     fd = df[5:]
#     fd.columns = col  # column 새로 지정
#     # # Date열을 제외한 모든 행의 값이 NaN인 경우 행 삭제
#     fd = fd.dropna(
#         subset=["0C1005/수정주가", "0C1008/거래량", "0C1009/거래대금", "0C1010/시가총액", "0C1013/외국인보유율", "0C1101/PER",
#                 "0C1103/PBR"],
#         how='all')
#
#     # print(fd.drop(labels = ['Unnamed: 0', 'Date'], axis=1))
#     fd2 = fd.drop(labels=['Date'], axis=1)
#
#     fd2['label'] = 0
#     fd2['len'] = len(fd2.index)
#     fd2 = fd2.reset_index(drop=True)
#     a = len(fd2.index)
#     # print(a)
#     if a < 274:
#         b = 275 - a
#         for i in range(b):
#             fd2.loc[a + i] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#     # print(len(fd2.index))
#     fd2 = fd2.fillna(0)
#     fd_f = fd2.values
#
#     input_data = fd_f[:, :-2]
#     input_data_norm = min_max_scaling(input_data)
#     # label = fd_f[1, -2]
#     label = [1,0]
#     len_stock = fd_f[1, -1]
#     # print(type(len_stock))
#     # label = label.tolist()
#     input_data = input_data.tolist()
#     # len_stock = len_stock.tolist()
#     result.append((input_data_norm, label, [len_stock]))
#         # print(result)
#
#     return result
