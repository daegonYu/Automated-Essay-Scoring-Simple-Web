from matplotlib import pyplot as plt
import numpy as np

def graph(mode, essay_point, mean, font_size):
    points = [essay_point, mean]    
    
    x = np.arange(2)
    names = [f'my {mode} point', f'{mode} mean']
    plt.xticks(x,names)
    plt.bar(x, points, color=['blue','green'])
    plt.ylim(0, max(points)+10)
    # 그래프에 텍스트 넣기
    for i, v in enumerate(x):
        plt.text(v, points[i], points[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
        fontsize = font_size, 
        color='black',
        horizontalalignment='center',  # horizontalalignment (left, center, right)
        verticalalignment='bottom')    # verticalalignment (top, center, bottom)

    plt.savefig(f'static/images/{mode}.png') 
    plt.show()    
    plt.clf()       # 그래프 초기화