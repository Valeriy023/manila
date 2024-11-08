import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition._pca import PCA
from scipy.stats import gaussian_kde
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
# Путь к файлу Excel
file_path = 'C:/manila/15 Гц, шаг 1.4, 3 выборка_ПРИМЕР.xlsx'

# Чтение данных из файла Excel
# словарь датафреймов
data = pd.read_excel(file_path, None)
data_norm = copy.deepcopy(data)
keys = list(data_norm.keys())

data_size1 = data_norm[keys[0]].shape[0]
data_size2 = data_norm[keys[0]].shape[1]

array = np.zeros([data_size1, data_size2, 3])
i = 0

# Деление каждой строки на ее нулевой элемент
for x in ["НР", "ЖТ", "ФЖ"]:
    data_norm[x] = data[x].apply(lambda row: row / row.iloc[0], axis=1)
    array[:, :, i] = data_norm[x].to_numpy()
    i = i + 1

# сделали столбец меток для каждого ритма (1 - НР, 2 - ЖТ, 3 - ФЖ)
array[:, 0, 1] = 2.0
array[:, 0, 2] = 3.0

# делим на тест и тейн только для К-БС
X_train = np.empty((0, data_size2))
X_test = np.empty((0, data_size2))
Y_train = np.empty((0, 1))
Y_test = np.empty((0, 1))

for o in range(3):
    X_train = np.vstack((X_train, array[:15, :, o]))
    X_test = np.vstack((X_test, array[15:, :, o]))

Y_train = X_train[:, 0]
Y_test = X_test[:, 0]

X_train = np.delete(X_train, 0, axis=1)
X_test = np.delete(X_test, 0, axis=1)

all_data = np.vstack((data_norm['НР'], data_norm['ЖТ'], data_norm['ФЖ']))

#к ближайших соседей
model = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

#оценка близости для k ближайших соседей
model2 = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean')
model2.fit(X_train, Y_train)
predictions2 = model2.predict(X_test)

#print(classification_report(Y_test, predictions))
#print(confusion_matrix(Y_test, predictions))

#print(classification_report(Y_test, predictions2))
#print(confusion_matrix(Y_test, predictions2))


#МГК чтобы посмотреть на разделяемость классов 
# Валера прости я все удалила, потому что начала путаться =(
# скаты нарисовала после оценки % дисп, да и вроде получилось поменьше кода

# САМОДЕЯТЕЛЬНОСТЬ МАШИ СТАРТ
# PCA для % объяснённой дисп
PCA_var = PCA(n_components=data_size2)  # число компонент = число признаков
PCA_var.fit(all_data)  # применение PCA к данным

# считаем процент объясненной дисперсии
explained_variance_ratio = PCA_var.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure()
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio * 100, marker='o', linestyle='-', color='orange')
plt.xlabel('Количество главных компонент')
plt.ylabel('Процент объясненной дисперсии')
plt.title('Процент объясненной дисперсии от числа главных компонент')
#plt.ylim(0, 110) #если очень хочется по красоте с нуля график
plt.grid()
plt.show()

# переделываю скат
PCA_scatter = PCA(n_components=2)
proj_data = PCA_scatter.fit_transform(all_data)

# строю скаты для первых 2х ГК
plt.figure(figsize=(8, 6))
plt.scatter(proj_data[:30, 0], proj_data[:30, 1], label='НР', alpha=0.7) # альфа тут типа ~прозрачность~ вау +вайб
plt.scatter(proj_data[30:60, 0], proj_data[30:60, 1], label='ЖТ', alpha=0.7)
plt.scatter(proj_data[60:, 0], proj_data[60:, 1], label='ФЖ', alpha=0.7)

plt.xlabel('I ГК')
plt.ylabel('II ГК')
plt.title('Объекты в пространстве первых двух главных компонент')
plt.legend()
plt.grid()
plt.show()

# Применение PCA к данным для получения первых трёх главных компонент
PCA_scatter_3d = PCA(n_components=3)
proj_data_3d = PCA_scatter_3d.fit_transform(all_data)

# Создание 3D-фигуры
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Построение трёхмерной скаттерограммы
ax.scatter(proj_data_3d[:30, 0], proj_data_3d[:30, 1], proj_data_3d[:30, 2], label='НР', alpha=0.7)
ax.scatter(proj_data_3d[30:60, 0], proj_data_3d[30:60, 1], proj_data_3d[30:60, 2], label='ЖТ', alpha=0.7)
ax.scatter(proj_data_3d[60:, 0], proj_data_3d[60:, 1], proj_data_3d[60:, 2], label='ФЖ', alpha=0.7)

# Настройки графика
ax.set_xlabel('1 ГК')
ax.set_ylabel('2 ГК')
ax.set_zlabel('3 ГК')
plt.title('Объекты в пространстве первых трёх главных компонент')
ax.legend()
plt.show()

#метод наименьших расстояний
# функция для объединения близких классов
def class_stack(classesData, key1:str, key2:str):
    class1 = classesData[key1].to_numpy()
    class2 = classesData[key2].to_numpy()
    stackedClasses = np.vstack((class1, class2)) #class1._append(class2, ignore_index = True)
    stackedClasses = np.delete(stackedClasses, 0, axis=1)
    return stackedClasses

# применение метода классификации по минимуму расстояния для 3 классов
stackedClasses = class_stack(data_norm, 'НР', 'ЖТ')
stackedClasses_means = np.mean(stackedClasses, axis=0)
shamefulClass = np.delete(data_norm['ФЖ'].to_numpy(), 0, axis=1)
shamefulClass_mean = np.mean(shamefulClass, axis=0)

w_min_dist = stackedClasses_means - shamefulClass_mean
w_norm_min_dist = w_min_dist/np.linalg.norm(w_min_dist)

projClass1 = np.matmul(stackedClasses, w_norm_min_dist)
projClass2 = np.matmul(shamefulClass, w_norm_min_dist)

kde1 = gaussian_kde(projClass1)
kde2 = gaussian_kde(projClass2)

# Определяем диапазон значений для оси X
x_vals = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 1000)

# Вычисляем значения плотности для обоих наборов данных
kde1_vals = kde1(x_vals)
kde2_vals = kde2(x_vals)

# Находим точки пересечения, где разница меняет знак
difference = kde1_vals - kde2_vals
sign_changes = np.where(np.diff(np.sign(difference)))[0]

# Отображаем гистограммы, огибающие и точки пересечения
plt.figure()
plt.hist(projClass1, bins=20, density=True, color='blue', edgecolor='black', alpha=0.5, label='1+2')
plt.hist(projClass2, bins=20, density=True, color='orange', edgecolor='black', alpha=0.5, label='3') 
plt.plot(x_vals, kde1_vals, color='red', linewidth=2, label="Огибающая 1+2")
plt.plot(x_vals, kde2_vals, color='blue', linewidth=2, label="Огибающая 3")
plt.xlim(left=min(projClass1.min(), projClass2.min()), right=max(projClass1.max(), projClass2.max()))
threshold = x_vals[sign_changes][0]
plt.axvline(x_vals[sign_changes], label=f'Порог: x= {threshold:.4f}')  # красные точки для пересечения
plt.xlabel('Значение')
plt.ylabel('Плотность')
plt.title('Гистограммы с пересечением огибающих')
plt.legend()
plt.show()

# применение метода классификации по минимуму расстояния для 2 классов
class1 = np.delete(data_norm['НР'].to_numpy(), 0, axis=1)
class2 = np.delete(data_norm['ЖТ'].to_numpy(), 0, axis=1)
class1_mean = np.mean(class1, axis=0)
class2_mean = np.mean(class2, axis=0)
w_min_dist_2 = class1_mean - class2_mean
w_norm_min_dist_2 = w_min_dist_2/np.linalg.norm(w_min_dist_2)

projClass1_2 = np.matmul(class1, w_norm_min_dist_2)
projClass2_2 = np.matmul(class2, w_norm_min_dist_2)

kde1_2 = gaussian_kde(projClass1_2)
kde2_2 = gaussian_kde(projClass2_2)

# Определяем диапазон значений для оси X
x_vals_2 = np.linspace(min(projClass1_2.min(), projClass2_2.min()), max(projClass1_2.max(), projClass2_2.max()), 1000)

# Вычисляем значения плотности для обоих наборов данных
kde1_vals_2 = kde1_2(x_vals_2)
kde2_vals_2 = kde2_2(x_vals_2)

# Находим точки пересечения, где разница меняет знак
difference_2 = kde1_vals_2 - kde2_vals_2
sign_changes_2 = np.where(np.diff(np.sign(difference_2)))[0]

# Отображаем гистограммы, огибающие и точки пересечения
plt.figure()
plt.hist(projClass1_2, bins=20, density=True, color='blue', edgecolor='black', alpha=0.5, label='1')
plt.hist(projClass2_2, bins=20, density=True, color='orange', edgecolor='black', alpha=0.5, label='2') 
plt.plot(x_vals_2, kde1_vals_2, color='red', linewidth=2, label="Огибающая 1")
plt.plot(x_vals_2, kde2_vals_2, color='blue', linewidth=2, label="Огибающая 2")
plt.xlim(left=min(projClass1_2.min(), projClass2_2.min()), right=max(projClass1_2.max(), projClass2_2.max()))
threshold_2 = x_vals_2[sign_changes_2][0]
plt.axvline(x_vals_2[sign_changes_2], label=f'Порог: x= {threshold_2:.4f}')  # красные точки для пересечения
plt.xlabel('Значение')
plt.ylabel('Плотность')
plt.title('Гистограммы с пересечением огибающих')
plt.legend()
plt.show()

# средние и дисперсии проекций
projClass1_M = np.mean(projClass1)
projClass2_M = np.mean(projClass2)
projClass1_M_2 = np.mean(projClass1_2)
projClass2_M_2 = np.mean(projClass2_2)
projClass1_D = np.var(projClass1)
projClass2_D = np.var(projClass2)
projClass1_D_2 = np.var(projClass1_2)
projClass2_D_2 = np.var(projClass2_2)

# построение ROC-кривой
all_data_2 = np.delete(all_data, 0, axis=1)
r1 = np.matmul(all_data_2, w_norm_min_dist) - threshold

ones = np.ones(30) * 1    # Массив из 30 единиц
twos = np.ones(30) * 2     # Массив из 30 двоек
threes = np.ones(30) * 3   # Массив из 30 троек
metki1_train = np.concatenate((ones, twos, threes))

def pizdato1(data, w, threshold, metki_train):
    r1 = np.matmul(data, w) - threshold

    metki1 = []
    for i in range(len(r1)):
        if r1[i] > 0:
            metki1.append(1)
        else:
            metki1.append(0)
    metki1_test = np.array(metki1)

    tp1 = 0
    tn1 = 0
    fp1 = 0
    fn1 = 0
    for i in range(len(metki1_test)):
        if metki_train[i] == 1 or metki_train[i] == 2:
            if metki1_test[i] == 1:
                tp1 += 1
            else:
                fn1 += 1
        else:
            if metki1_test[i] == 1:
                fp1 += 1
            else:
                tn1 += 1
    se1 = tp1/(tp1+fn1)
    sp1 = tn1/(tn1+fp1)
    acc1 = (tn1+tp1)/(tn1+tp1+fn1+fp1)
    return se1, sp1, acc1

se1 = []
sp1 = []
thresholds1 = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 100)
for i in thresholds1:
    se, sp, _ = pizdato1(all_data_2, w_min_dist, i, metki1_train)
    se1.append(se)
    sp1.append(sp*(-1) + 1)

plt.figure()
plt.plot(sp1, se1)
'''
r2 = np.matmul(stackedClasses, w_norm_min_dist_2) - threshold_2

metki2_train = np.concatenate((ones, twos))

metki2 = []
for i in range(len(r2)):
    if r2[i] > 0:
        metki2.append(1)
    else:
        metki2.append(2)
metki2_test = np.array(metki2)

tp2 = 0
tn2 = 0
fp2 = 0
fn2 = 0
for i in range(len(metki2_test)):
    if metki2_train[i] == 2:
        if metki2_test[i] == 2:
            tp2 += 1
        else:
            fn2 += 1
    else:
        if metki2_test[i] == 1:
            tn2 += 1
        else:
            fp2 += 1
print(tp2, tn2, fp2, fn2)
print(metki2_test)



se2 = tp2/(tp2+fn2)
sp2 = tn2/(tn2+fp2)
acc2 = (tn2+tp2)/(tn2+tp2+fn2+fp2)
'''
# Параметры гауссовых распределений для двух классов
mu1, sigma1 = projClass1_M, projClass1_D  # Класс 1
mu0, sigma0 = projClass2_M, projClass2_D  # Класс 0

# Пороги для классификации
thresholds = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 1000)
tpr = []
fpr = []

for T in thresholds:
    # Рассчитаем TPR и FPR для текущего порога
    tpr_value = 1 - norm.cdf(T, mu1, sigma1)  # Интеграл для класса 1
    fpr_value = 1 - norm.cdf(T, mu0, sigma0)  # Интеграл для класса 0
    
    tpr.append(tpr_value)
    fpr.append(fpr_value)

# Построение ROC-кривой
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая методом Гаусса")
plt.legend()
plt.show()

'''
sns.histplot(projClass1, kde=True, bins=20, color='skyblue', edgecolor='black')
sns.histplot(projClass2, kde=True, bins=20, color='purple', edgecolor='black')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.title('Гистограмма с оценкой плотности KDE')
'''

#ЛДФ- Фишер, сначала посмотрели МГК, выделили "позорный класс"

# САМОДЕЯТЕЛЬНОСТЬ МАШИ КОНЕЦ
'''
fig, axs = plt.subplots(3, 1)

diff_mean = np.mean(X_train[:15, :], axis=0) - np.mean(X_train[15:45, :], axis=0)
sum_covariance = np.cov(X_train[:15, :], rowvar=0) + np.cov(X_train[15:45, :], rowvar=0)
W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
w = W/np.linalg.norm(W)
proj_class1 = np.matmul(X_train[:15, :], w)
proj_class2 = np.matmul(X_train[15:45, :], w)

hist_data = np.vstack((proj_class1, proj_class2[0:15], proj_class2[15:]))
axs[0].hist(hist_data.T, bins= 20, edgecolor = 'black', label=['1', '2', '3'])
axs[0].set_title('1 и 2+3')
axs[0].legend(loc='best')

diff_mean = np.mean(X_train[30:45, :], axis=0) - np.mean(X_train[:30, :], axis=0)
sum_covariance = np.cov(X_train[30:45, :], rowvar=0) + np.cov(X_train[:30, :], rowvar=0)
W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
w = W/np.linalg.norm(W)
proj_class1 = np.matmul(X_train[30:45, :], w)
proj_class2 = np.matmul(X_train[:30, :], w)

#np.reshape(proj_class2)

hist_data = np.vstack((proj_class1, proj_class2[0:15], proj_class2[15:]))
axs[1].hist(hist_data.T, bins= 20, edgecolor = 'black', label=['3', '1', '2'])
axs[1].set_title('1+2 и 3')
axs[1].legend(loc='best')

X_train13 = np.vstack((X_train[0:15, :], X_train[30:45, :]))

diff_mean = np.mean(X_train13, axis=0) - np.mean(X_train[15:30, :], axis=0)
sum_covariance = np.cov(X_train13, rowvar=0) + np.cov(X_train[15:30, :], rowvar=0)
W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
w = W/np.linalg.norm(W)
proj_class1 = np.matmul(X_train13, w)
proj_class2 = np.matmul(X_train[15:30, :], w)

hist_data = np.vstack((proj_class1[:15], proj_class1[15:30], proj_class2))
axs[2].hist(hist_data.T, bins= 20, edgecolor = 'black', label=['1', '3', '2'])
axs[2].set_title('1+3 и 2')
axs[2].legend(loc='best')
plt.show()

fig, axs = plt.subplots(3, 1)

diff_mean = np.mean(X_train[:15, :], axis=0) - np.mean(X_train[15:30, :], axis=0)
sum_covariance = np.cov(X_train[:15, :], rowvar=0) + np.cov(X_train[15:30, :], rowvar=0)
W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
w = W/np.linalg.norm(W)
proj_class1 = np.matmul(X_train[:15, :], w)
proj_class2 = np.matmul(X_train[15:30, :], w)

hist_data = np.vstack((proj_class1, proj_class2))
axs[0].hist(hist_data.T, bins= 20, edgecolor = 'black', label=['1', '2'])
axs[0].set_title('1 и 2')
axs[0].legend(loc='best')

diff_mean = np.mean(X_train[15:30, :], axis=0) - np.mean(X_train[30:45, :], axis=0)
sum_covariance = np.cov(X_train[15:30, :], rowvar=0) + np.cov(X_train[30:45, :], rowvar=0)
W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
w = W/np.linalg.norm(W)
proj_class1 = np.matmul(X_train[15:30, :], w)
proj_class2 = np.matmul(X_train[30:45, :], w)

#np.reshape(proj_class2)

hist_data = np.vstack((proj_class1, proj_class2))
axs[1].hist(hist_data.T, bins= 20, edgecolor = 'black', label=['2', '3'])
axs[1].set_title('2 и 3')
axs[1].legend(loc='best')

diff_mean = np.mean(X_train[:15, :], axis=0) - np.mean(X_train[30:45, :], axis=0)
sum_covariance = np.cov(X_train[:15, :], rowvar=0) + np.cov(X_train[30:45, :], rowvar=0)
W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
w = W/np.linalg.norm(W)
proj_class1 = np.matmul(X_train[:15, :], w)
proj_class2 = np.matmul(X_train[30:45, :], w)

hist_data = np.vstack((proj_class1, proj_class2))
axs[2].hist(hist_data.T, bins= 20, edgecolor = 'black', label=['1', '3'])
axs[2].set_title('1 и 3')
axs[2].legend(loc='best')
plt.show()

#порог ищите каждый сам!!!

#МДА
covNR = np.cov(X_train[:15, :], rowvar=0)
covJT = np.cov(X_train[15:30, :], rowvar=0)
covFJ = np.cov(X_train[30:45, :], rowvar=0)
'''