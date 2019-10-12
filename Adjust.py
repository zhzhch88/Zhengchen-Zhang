import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xlrd
################### 各种回归算法包 ########################
from sklearn.linear_model import LinearRegression   ## 多元线性回归
from sklearn import tree ##  决策树
from sklearn import svm  ## SVM回归
from sklearn import neighbors  ## KNN回归
from sklearn import ensemble  ##随机森林

#############   样本库建立   ##################
path = 'x:/xx/xx.xls'
save_path = 'x:/xx/'
book = xlrd.open_workbook(path)
table = book.sheets()[0]
nr = table.nrows
target = table.col_values(3)[:]   ##可改
data = []
for ax in range(, ):
    data_row = table.row_values(ax)[:]
    data.append(data_row)

def model_study(data,target):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    ##print(np.array(data))   ##查看样本矩阵
    ##print(np.array(y_test))

    #model = LinearRegression()   ## 线性回归
    ###################其他的一些回归算法###########################
    #model = tree.DecisionTreeRegressor(max_depth=3)  ## tree   最大深度预防过拟合
    model = svm.SVR(gamma=1,kernel='linear',C=10)  ## 指定要在算法中使用的内核类型。
                                                ## 它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'
                                                ## 或者callable之一。如果没有给出，将使用'rbf'。
    #model = neighbors.KNeighborsRegressor()    ##  KNN
    #model = ensemble.RandomForestRegressor(n_estimators=4,min_samples_leaf=5)     ## n_estimators代表有几棵树

    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    print(score)
    result = model.predict(x_test)
    print(model.coef_)    ## 输出权值   线性可推出公式
    #print(model.intercept_)  ## 输出偏值，在线性回归中代表截距
    #print(model.feature_importances_)   ##输出权值，但是该属性不适用于线性回归

    ###############     用学习好的模型来预测值    ########################
    ##predict_value = model.predict(np.array([[3,5,7]]))
    ##print(predict_value)

    ##### 绘图
    plt.figure()
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score: %f' % score)
    plt.legend()   ## 用于显示图例
    plt.show()
    # plt.savefig(save_path + 'picture.png', bbox_inch='tight')

scores = model_study(data, target)
