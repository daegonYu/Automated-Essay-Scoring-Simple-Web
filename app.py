from flask import Flask, render_template, request
from forms import DocumentBertScoringModel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re

app = Flask(__name__)

config_ = '/home/daegon/AES/models/chunk_model.bin1/config.json'    # config는 모두 같다.

chunk_model_path = '/home/daegon/AES/models/chunk_model.bin1'; word_doc_model_path = '/home/daegon/AES/models/word_doc_model.bin1' 
logical_model = DocumentBertScoringModel(chunk_model_path= chunk_model_path, word_doc_model_path= word_doc_model_path, config= config_)

chunk_model_path = '/home/daegon/AES/models/chunk_model.bin2'; word_doc_model_path = '/home/daegon/AES/models/word_doc_model.bin2' 
novelty_model = DocumentBertScoringModel(chunk_model_path= chunk_model_path, word_doc_model_path= word_doc_model_path, config= config_)

chunk_model_path = '/home/daegon/AES/models/chunk_model.bin3'; word_doc_model_path = '/home/daegon/AES/models/word_doc_model.bin3' 
persuasive_model = DocumentBertScoringModel(chunk_model_path= chunk_model_path, word_doc_model_path= word_doc_model_path, config= config_)

# 경로 확인 필요
logical_hub_points = np.load('./aihub_point/logical.npy')
novelty_hub_points = np.load('./aihub_point/novelty.npy')
persuasive_hub_points = np.load('./aihub_point/persuasive.npy')

logical_mean = logical_hub_points.mean()
novelty_mean = novelty_hub_points.mean()
persuasive_mean = persuasive_hub_points.mean()

logical_mean = round(float(logical_mean),2)
novelty_mean = round(float(novelty_mean),2)
persuasive_mean = round(float(persuasive_mean),2)

mean_list = [logical_mean,novelty_mean,persuasive_mean]

logical_hub_points_list = list(logical_hub_points)
novelty_hub_points_list = list(novelty_hub_points)
persuasive_hub_points_list = list(persuasive_hub_points)


x = np.arange(2)

# title_font = {        # plt
#     'fontsize': 16,
#     'fontweight': 'bold'
# }

font_size = 20

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        essay = request.form["essay"]
        essay = re.sub('\n|#|"', ' ', essay)        # 점수 오류는 여기 밖에 없지 이 변수 밖에 없어
        print(essay)
        input_sentence = [essay,""]
        logical_point = logical_model.result_point(input_sentence, mode_='logical')
        novelty_point = novelty_model.result_point(input_sentence, mode_='novelty')
        persuasive_point = persuasive_model.result_point(input_sentence, mode_='persuasive')
        total_point = logical_point+novelty_point+persuasive_point
        
        point_list = [logical_point, novelty_point, persuasive_point]
        
        logical_hub_points_list.append(logical_point)
        novelty_hub_points_list.append(novelty_point)
        persuasive_hub_points_list.append(persuasive_point)
        
        logical_hub_points_list.sort(); novelty_hub_points_list.sort(); persuasive_hub_points_list.sort()
        
        # 점수가 상위 몇 % 인지 계산
        my_logical_grade = (len(logical_hub_points_list) - logical_hub_points_list.index(logical_point)) / len(logical_hub_points_list) * 100
        my_novelty_grade = (len(novelty_hub_points_list) - novelty_hub_points_list.index(novelty_point)) / len(novelty_hub_points_list) * 100
        my_persuasive_grade = (len(persuasive_hub_points_list) - persuasive_hub_points_list.index(persuasive_point)) / len(persuasive_hub_points_list) * 100
        
        grade_list = [my_logical_grade, my_novelty_grade, my_persuasive_grade]     
        grade_list = [round(grade,2) for grade in grade_list]
        
        points1 = [logical_point, logical_mean]
        names = ['my logical point', 'logical mean']
        # plt.title("logical point")
        plt.xticks(x,names)
        plt.bar(x, points1, color=['blue','green'])
        plt.ylim(0, max(points1)+10)
        # 그래프에 텍스트 넣기
        for i, v in enumerate(x):
            plt.text(v, points1[i], points1[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
            fontsize = font_size, 
            color='black',
            horizontalalignment='center',  # horizontalalignment (left, center, right)
            verticalalignment='bottom')    # verticalalignment (top, center, bottom)

        plt.savefig('static/images/logical.png') 
        plt.show()    
        plt.clf()       # 그래프 초기화
        
        
        points2 = [novelty_point, novelty_mean]
        names = ['my novelty point', 'novelty mean']
        # plt.title("novelty point")
        plt.xticks(x,names)
        plt.bar(x, points2, color=['blue','green'])
        plt.ylim(0, max(points2)+10)
        for i, v in enumerate(x):
            plt.text(v, points2[i], points2[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
            fontsize = font_size, 
            color='black',
            horizontalalignment='center',  # horizontalalignment (left, center, right)
            verticalalignment='bottom')    # verticalalignment (top, center, bottom)
        
        plt.savefig('static/images/novelty.png')   
        plt.show()
        plt.clf()
        
        points3 = [persuasive_point, persuasive_mean]
        names = ['my persuasive point', 'persuasive mean'] 
        # plt.title("persuasive point")
        plt.xticks(x,names)
        plt.bar(x, points3, color=['blue','green'])
        plt.ylim(0, max(points3)+10)
        for i, v in enumerate(x):
            plt.text(v, points3[i], points3[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
            fontsize = font_size, 
            color='black',
            horizontalalignment='center',  # horizontalalignment (left, center, right)
            verticalalignment='bottom')    # verticalalignment (top, center, bottom)
       
        plt.savefig('static/images/persuasive.png')   
        plt.show()
        plt.clf()
        
        return render_template("result.html", essay=essay, point_list=point_list, \
            total_point=total_point, grade_list=grade_list, mean_list=mean_list)
    else:
        return render_template("index.html")



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5502)