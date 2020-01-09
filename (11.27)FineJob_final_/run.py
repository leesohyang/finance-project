from flask import Flask, render_template, request
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from att_lstm_test import *
from temp_test import *
import financepage as fp
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
import json
import financepage as fp

app = Flask(__name__) # 매개변수 __name__으로 새로운 플라스크 인스턴스를 초기화
company= ''
@app.route('/')
def index():
   return render_template('searchpage.html')

@app.route('/company', methods=['GET','POST'])
def company():
    if request.method == 'POST':
        review = request.form['review'] #name=복지가 좋은 기업 추천해줘
        locate = request.form['location']
        rule = request.form['rule']
        data = pd.read_csv(r"C:\Users\이소향\Downloads\abc.csv")
        company_label = test(review)

        com_ls = data['기업']
        label_ls = data['label']
        rate_ls = data['평점']
        locate_ls = data['지역']
        rule_ls = data['직무']

        data_2 = pd.DataFrame()
        data_2['기업'] = com_ls
        data_2['label'] = label_ls
        data_2['평점'] = rate_ls
        data_2['지역'] = locate_ls
        data_2['직무'] = rule_ls

        data_2.index = label_ls

        # data_2.drop_duplicates()
        data_final = data_2.loc[company_label].drop_duplicates()
        locate_ls_2 = data_final['지역']
        data_final.index = locate_ls_2
        data_final = data_final.loc[locate].drop_duplicates()
        rule_ls_2 = data_final['직무']
        data_final.index = rule_ls_2
        data_final = data_final.loc[rule].drop_duplicates()
        data_fin = data_final.sort_values(by=['평점'], ascending=False)[:10]

        comp_list = data_fin['기업'].tolist()
        rate_list = data_fin['평점'].tolist()
        # 위에까지 기업 추천
        json_file = open('static/interview.json').read()
        jsondata = json.loads(json_file)

        rating_json = json.loads(open('static/평점.json', encoding="UTF-8").read())
        data_json = json.loads(open('static/기업정보.json', encoding="UTF-8").read())
        cloud_json = json.loads(open('static/cloud.json', encoding="UTF-8").read())
        review_json = json.loads(open('static/review.json', encoding="UTF-8").read())
        interview_json = json.loads(open('static/면접.json', encoding="UTF-8").read())

        # review = request.form['review']
        # name1 = fp.getName(review)  # LSTM 결과 받은 기업명들
        # name2 = "버즈빌(주)"
        name1 = "이녹스(주)"

        ls_r = fp.getReview(name1)
        # ls_r = fp.getReview(name1)

        return render_template("company.html", review = review, name0 = comp_list[0], name1 = comp_list[1], name2 = comp_list[2],
                             name3 = comp_list[3], name4 = comp_list[4], name5 = comp_list[5], name6 = comp_list[6],
                             name7 = comp_list[7], name8 = comp_list[8], name9 = comp_list[9],
                             rate0=rate_list[0], rate1=rate_list[1], rate2=rate_list[2],
                             rate3=rate_list[3], rate4=rate_list[4], rate5=rate_list[5], rate6=rate_list[6],
                             rate7=rate_list[7], rate8=rate_list[8], rate9=rate_list[9],
                             jsondata=json.dumps(jsondata),
                             rating_json=json.dumps(rating_json),
                             data_json=json.dumps(data_json),
                             cloud_json=json.dumps(cloud_json),
                             review_json=json.dumps(review_json),
                             interview_json=json.dumps(interview_json),

                               ls_r=ls_r,
                             ) #여기다가 변수 입력.

@app.route('/finance', methods=['GET','POST'])
def finance():
    if request.method == "GET":
        company = request.args.get('company')
        dataIdx_dic = {"이녹스(주)": "088390"}

        company_code = dataIdx_dic[company] #이걸 financepage.py에 전달
        # business_Summary
        Summary1, Summary2 = fp.makeBusinessSummary(company) #Summary2만 써
        mydfs = fp.searchAndMakeDataframe(company)
        graph_sale = fp.makesalegraph(company_code)
        graph_stock = fp.makeCompanyFinance_stock(company)
        # 상장폐지예측모델
        predict = test_finanace(company)
        # print('finance 탭 클릭')
        return render_template("finance.html", name=company, accuracy=int(predict * 100), Summary=Summary2, tables2=mydfs[2],
                               # en_name=mydfs[1].iloc[0].to_html)
                               tables0=mydfs[0], tables1=mydfs[1], graph_sale=graph_sale, graph_stock=graph_stock)
    else:
        return render_template("finance.html")


if __name__ == '__main__':
   app.run(debug = True)
