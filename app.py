from flask import Flask, render_template, request
from forms import DocumentBertScoringModel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

app = Flask(__name__)

config_ = '/home/daegon/AES/models/chunk_model.bin1/config.json'    # config는 모두 같다.

chunk_model_path = '/home/daegon/AES/models/chunk_model.bin1'; word_doc_model_path = '/home/daegon/AES/models/word_doc_model.bin1' 
logical_model = DocumentBertScoringModel(chunk_model_path= chunk_model_path, word_doc_model_path= word_doc_model_path, config= config_)

chunk_model_path = '/home/daegon/AES/models/chunk_model.bin2'; word_doc_model_path = '/home/daegon/AES/models/word_doc_model.bin2' 
novelty_model = DocumentBertScoringModel(chunk_model_path= chunk_model_path, word_doc_model_path= word_doc_model_path, config= config_)

chunk_model_path = '/home/daegon/AES/models/chunk_model.bin3'; word_doc_model_path = '/home/daegon/AES/models/word_doc_model.bin3' 
persuasive_model = DocumentBertScoringModel(chunk_model_path= chunk_model_path, word_doc_model_path= word_doc_model_path, config= config_)

logical_hub_points = np.load('./aihub_point/logical.npy')
novelty_hub_points = np.load('./aihub_point/novelty.npy')
persuasive_hub_points = np.load('./aihub_point/persuasive.npy')

logical_mean = logical_hub_points.mean()
novelty_mean = novelty_hub_points.mean()
persuasive_mean = persuasive_hub_points.mean()

x = np.arange(2)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        essay = request.form["essay"]
        print(essay)
        input_sentence = [essay,""]
        logical_point = logical_model.result_point(input_sentence, mode_='logical')
        novelty_point = novelty_model.result_point(input_sentence, mode_='novelty')
        persuasive_point = persuasive_model.result_point(input_sentence, mode_='persuasive')
        total_point = logical_point+novelty_point+persuasive_point
        
        
        points1 = [logical_point, round(float(logical_mean),2)]
        names = ['my logical point', 'logical mean']
        plt.xticks(x,names)
        plt.bar(x, points1)
        plt.savefig('static/images/logical.png')     
        
        
        points2 = [novelty_point, round(float(novelty_mean),2)]
        names = ['my novelty point', 'novelty mean']
        plt.xticks(x,names)
        plt.bar(x, points2)
        plt.savefig('static/images/novelty.png')   
        
        
        points3 = [persuasive_point, round(float(persuasive_mean),2)]
        names = ['my persuasive point', 'persuasive mean'] 
        plt.xticks(x,names)
        plt.bar(x, points3)
        plt.savefig('static/images/persuasive.png')   
        
        print("사진생성")
        
        return render_template("result.html", essay=essay, logical_point=logical_point, novelty_point=novelty_point, persuasive_point=persuasive_point, total_point=total_point, n=1)
    else:
        return render_template("index.html")



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5502)