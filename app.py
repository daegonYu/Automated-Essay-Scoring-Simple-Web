from flask import Flask, render_template, request
from forms import DocumentBertScoringModel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import pic

app = Flask(__name__)

# 모델 불러오기
config_ = 'models/chunk_model.bin10/config.json'    # config는 모두 같다.

chunk_model_path = 'models/logical_finished_chunk_model.bin1'; word_doc_model_path = 'models/logical_finished_word_doc_model.bin1' 
logical_model = DocumentBertScoringModel(chunk_model_path=chunk_model_path, word_doc_model_path=word_doc_model_path, config=config_)

chunk_model_path = 'models/reason_finished_chunk_model.bin1'; word_doc_model_path = 'models/reason_finished_word_doc_model.bin1' 
reason_model = DocumentBertScoringModel(chunk_model_path=chunk_model_path, word_doc_model_path=word_doc_model_path, config=config_)

chunk_model_path = 'models/persuasive_finished_chunk_model.bin1'; word_doc_model_path = 'models/persuasive_finished_word_doc_model.bin1' 
persuasive_model = DocumentBertScoringModel(chunk_model_path=chunk_model_path, word_doc_model_path=word_doc_model_path, config=config_)

chunk_model_path = 'models/novelty_finished_chunk_model.bin1'; word_doc_model_path = 'models/novelty_finished_word_doc_model.bin1' 
novelty_model = DocumentBertScoringModel(chunk_model_path=chunk_model_path, word_doc_model_path=word_doc_model_path, config=config_)

# 경로 확인 필요
logical_hub_points = np.load('./aihub_point/logical.npy')
reason_hub_points = np.load('./aihub_point/reason.npy')
persuasive_hub_points = np.load('./aihub_point/persuasive.npy')
novelty_hub_points = np.load('./aihub_point/novelty.npy')

logical_mean = logical_hub_points.mean()
reason_mean = reason_hub_points.mean()
persuasive_mean = persuasive_hub_points.mean()
novelty_mean = novelty_hub_points.mean()

# (53.31, 59.04, 53.02, 50.98)
logical_mean = round(float(logical_mean),2)
reason_mean = round(float(reason_mean),2)
persuasive_mean = round(float(persuasive_mean),2)
novelty_mean = round(float(novelty_mean),2)

total_score_per_student = logical_hub_points+reason_hub_points+persuasive_hub_points+novelty_hub_points
total_score_per_student /= 4        # 평균
total_score_per_student = np.round(total_score_per_student, 2)  # 반올림

hub_points = [logical_mean, reason_mean, persuasive_mean, novelty_mean]
hub_mean = round(sum(hub_points) / len(hub_points),2)

# numpy -> list
logical_hub_points_list = list(logical_hub_points)
reason_hub_points_list = list(reason_hub_points)
persuasive_hub_points_list = list(persuasive_hub_points)
novelty_hub_points_list = list(novelty_hub_points)
total_score_per_student = list(total_score_per_student)

# title_font = {        # plt
#     'fontsize': 16,
#     'fontweight': 'bold'
# }


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        # 에세이 텍스트 박스로 받으면 정제
        essay = request.form["essay"]
        essay = re.sub('\n|#|"', ' ', essay)        # \n, #, " 삭제
        print(essay)
        # 정제된 에세이를 모델에 입력
        input_sentence = [essay,""]
        logical_point = logical_model.result_point(input_sentence, mode_='logical')
        reason_point = reason_model.result_point(input_sentence, mode_='reason')
        persuasive_point = persuasive_model.result_point(input_sentence, mode_='persuasive')
        novelty_point = novelty_model.result_point(input_sentence, mode_='novelty')
        
        
        my_point_list = [logical_point, reason_point, persuasive_point, novelty_point]
        my_point_mean = round(sum(my_point_list) / len(my_point_list),2)
        total_mean = my_point_mean, hub_mean
        
        np_my_points = np.array(my_point_list)
        np_hub_points = np.array(hub_points)
        
        # boolean
        np_my_class = np_my_points > np_hub_points        # 내 점수가 더 높다면 True
        list_my_class =list(np_my_class)
        
        # list_my_clss[4] == True    -->  모든 요소에서 나의 점수가 평균보다 높다.
        if list_my_class.count(False) == 0:
            list_my_class.append(True)
        else:
            list_my_class.append(False)
        
        dic_my_class = {
            "논리성" : np_my_class[0],      # boolean
            "근거의 풍부함" : np_my_class[1],
            "설득력" : np_my_class[2],
            "참신성" : np_my_class[3],
        }
        my_class = dic_my_class, list_my_class[4]
        
        np_elements = np.array(list(dic_my_class.keys()))
        big_small_element = np_elements[[np.argmax(np_my_points), np.argmin(np_my_points)]]
        
        # 점수가 다른 학생들과 비교하여 상위 몇 % 인지 계산
        
        logical_hub_points_list.append(logical_point)
        reason_hub_points_list.append(reason_point)
        persuasive_hub_points_list.append(persuasive_point)
        novelty_hub_points_list.append(novelty_point)
        total_score_per_student.append(my_point_mean)
        
        logical_hub_points_list.sort(); reason_hub_points_list.sort(); persuasive_hub_points_list.sort(); novelty_hub_points_list.sort(); total_score_per_student.sort()
        
        # 순위 계산할 때는 round()가 적용되면 안되고 순위계산 후 보여줄 때 round()가 계산되어야 한다.
        my_logical_grade = (len(logical_hub_points_list) - logical_hub_points_list.index(logical_point)) / len(logical_hub_points_list) * 100
        my_reason_grade = (len(reason_hub_points_list) - reason_hub_points_list.index(reason_point)) / len(reason_hub_points_list) * 100
        my_persuasive_grade = (len(persuasive_hub_points_list) - persuasive_hub_points_list.index(persuasive_point)) / len(persuasive_hub_points_list) * 100
        my_novelty_grade = (len(novelty_hub_points_list) - novelty_hub_points_list.index(novelty_point)) / len(novelty_hub_points_list) * 100
        my_total_grade = (len(total_score_per_student) - total_score_per_student.index(my_point_mean)) / len(total_score_per_student) * 100
        my_total_grade = round(my_total_grade)
        
        grade_list = [my_logical_grade, my_reason_grade, my_persuasive_grade, my_novelty_grade]     
        grade_list = [round(grade) for grade in grade_list]
           
        # 그래프 그리고 저장하기
        
        font_size = 24
        
        pic.graph(mode='logical', essay_point=my_point_list[0], mean=logical_mean, font_size=font_size)
        pic.graph(mode='reason', essay_point=my_point_list[1], mean=reason_mean, font_size=font_size)
        pic.graph(mode='persuasive', essay_point=my_point_list[2], mean=persuasive_mean, font_size=font_size)
        pic.graph(mode='novelty', essay_point=my_point_list[3], mean=novelty_mean, font_size=font_size)
        pic.total_graph(mean=total_mean, my_points=my_point_list, hub_points=hub_points)
        
        return render_template("result.html", essay=essay, my_points=my_point_list, \
            hub_points=hub_points, grade_list=grade_list, total_mean=total_mean, my_total_grade=my_total_grade,\
                my_class = my_class, big_small_element=big_small_element)
    else:
        return render_template("index.html")



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5502)