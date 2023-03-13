import pandas as pd
import numpy as np
 
from math import pi

from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def graph(mode, essay_point, mean, font_size):
    my_palette = plt.cm.get_cmap("Set1", 2)
    
    points = [essay_point, mean]    
    x = np.arange(2)
    names = ['My point', 'Another student']
    plt.xticks(x,names, fontsize = font_size)
    plt.bar(x, points, color=[my_palette(0),my_palette(1)])
    plt.ylim(max(min(points)-25,0), max(points)+10)
    # 그래프에 텍스트 넣기
    for i, v in enumerate(x):
        plt.text(v, points[i], points[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
        fontsize = font_size, 
        color='black',
        horizontalalignment='center',  # horizontalalignment (left, center, right)
        verticalalignment='bottom')    # verticalalignment (top, center, bottom)

    plt.savefig(f'static/images/{mode}.png') 
    # plt.show()    
    plt.clf()       # 그래프 초기화
    

def total_graph(mean,my_points, hub_points):
    ## 데이터 준비
    df = pd.DataFrame({
    'Character': ['My Points','Another students'],
    'Mean point': [mean[0], mean[1]],
    'Logicality': [my_points[0], hub_points[0]],
    'Reasonable': [my_points[1], hub_points[1]],
    'Persuative': [my_points[2], hub_points[2]],
    'Novelty': [my_points[3], hub_points[3]]
    })
    labels = df.columns[1:]
    num_labels = len(labels)
        
    angles = [x/float(num_labels)*(2*pi) for x in range(num_labels)] ## 각 등분점
    angles += angles[:1] ## 시작점으로 다시 돌아와야하므로 시작점 추가
        
    my_palette = plt.cm.get_cmap("Set1", len(df.index))
    
    fig = plt.figure(figsize=(8,8))
    fig.set_facecolor('white')
    ax = fig.add_subplot(polar=True)
    for i, row in df.iterrows():
        color = my_palette(i)
        data = df.iloc[i].drop('Character').tolist()
        data += data[:1]
        
        ax.set_theta_offset(pi / 2) ## 시작점
        ax.set_theta_direction(-1) ## 그려지는 방향 시계방향
        
        plt.xticks(angles[:-1], labels, fontsize=15) ## x축 눈금 라벨
        ax.tick_params(axis='x', which='major', pad=15) ## x축과 눈금 사이에 여백을 준다.
        ax.set_rlabel_position(0) ## y축 각도 설정(degree 단위)
        plt.yticks([pr*10 for pr in range(0,11,2)],[str(pr*10) for pr in range(0,11,2)], fontsize=10) ## y축 눈금 설정
        plt.ylim(0,100)
        
        ax.plot(angles, data, color=color, linewidth=2, linestyle='solid', label=row.Character) ## 레이더 차트 출력
        ax.fill(angles, data, color=color, alpha=0.4) ## 도형 안쪽에 색을 채워준다.
        
    for g in ax.yaxis.get_gridlines(): ## grid line 
        g.get_path()._interpolation_steps = len(labels)
    
    spine = Spine(axes=ax,
            spine_type='circle',
            path=Path.unit_regular_polygon(len(labels)))
    
    ## Axes의 중심과 반지름을 맞춰준다.
    spine.set_transform(Affine2D().scale(.5).translate(.5, .5)+ax.transAxes)
            
    ax.spines = {'polar':spine} ## frame의 모양을 원에서 폴리곤으로 바꿔줘야한다.
    
    plt.legend(loc=(0.7, 1), fontsize=15)
    plt.savefig('static/images/total.png')
    # plt.show()
    plt.clf()       # 그래프 초기화
    