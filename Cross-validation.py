import xlrd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   ## 多元线性回归
from sklearn import tree ##  决策树
from sklearn import svm  ## SVM回归
from sklearn import neighbors  ## KNN回归
from sklearn import ensemble  ##随机森林

def regression(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    #model = LinearRegression()
    #model = tree.DecisionTreeRegressor(max_depth=4)  ## tree   最大深度预防过拟合
    #model = svm.SVR(gamma="auto",kernel="linear")  ## 指定要在算法中使用的内核类型
    model = neighbors.KNeighborsRegressor()    ##  KNN
    #model = ensemble.RandomForestRegressor(n_estimators=4,min_samples_leaf=5)     ## n_estimators代表有几棵树

    model.fit(x_train,y_train)
    score = cross_val_score(model, x_train, y_train, cv=5)
    print(score)
    return score


path = 'C:/Users/windows/Desktop/3.31/给师兄的数据改.xls'
save_path = 'C:/Users/windows/Desktop/'
book = xlrd.open_workbook(path)
table = book.sheets()[0]
target = table.col_values(3)[1:27]
nr = table.nrows
data = []
for ax in range(1, 27):
    data_row = table.row_values(ax)[19:]
    data.append(data_row)

# 机器学习交叉验证
score = regression(data, target)

scores = []
for i in range(len(score)):
        if score[i] > -1 and score[i]<1:
            scores.append(abs(score[i]))
        for param_name in target.params.keys():
            weight = score.params[param_name][0].data
        pf.write('\n' + param_name + '_weight:\n\n')
        weight.shape = (-1, 1)

        for w in weight:
            pf.write('%ff, ' % w)

print(scores)
print(np.mean(scores))


# 十折交叉验证绘图

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 画布大小设置
fig = plt.figure(figsize=(20, 9), dpi=300, facecolor='w', edgecolor='k')

plt.plot(range(len(scores)), scores, 'o-', color='red')
ax = plt.gca()
height = ax.get_ylim()[1] - ax.get_ylim()[0]
plt.tick_params(labelsize=16)

for a, b in zip(range(len(scores)), scores):
    plt.text(a, (b+height/50), '%0.5f' % b, ha='center', va='bottom', fontsize=16)

plt.axhline(np.mean(scores), color='red', ls='--')
plt.text(0, (np.mean(scores)+height/50), '%0.5f' % (np.mean(scores)+height/50), ha='center', va='bottom', fontsize=16, color='red')

plt.title('交叉验证准确率', fontsize=20)
plt.xticks(range(len(scores)))
plt.xlabel('次数', fontsize=20)
plt.ylabel('精度', fontsize=20)

# plt.tight_layout()

plt.savefig(save_path + '交叉验证随机森林的准确率.png', bbox_inch='tight')
