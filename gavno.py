import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition._pca import PCA
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, auc
from statsmodels.multivariate.manova import MANOVA
import seaborn as sns

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

'''
k ближайших соседей
'''
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

'''
МГК чтобы посмотреть на разделяемость классов
'''
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

'''
метод наименьших расстояний
'''
# функция для объединения близких классов
def class_stack(classesData, key1:str, key2:str):
    class1 = classesData[key1].to_numpy()
    class2 = classesData[key2].to_numpy()
    stackedClasses = np.vstack((class1, class2))
    stackedClasses = np.delete(stackedClasses, 0, axis=1)
    return stackedClasses

# применение метода классификации по минимуму расстояния для 3 классов - эти же классы use в ЛДФ
stackedClasses = class_stack(data_norm, 'НР', 'ЖТ')
stackedClasses_means = np.mean(stackedClasses, axis=0)
shamefulClass = np.delete(data_norm['ФЖ'].to_numpy(), 0, axis=1)
shamefulClass_mean = np.mean(shamefulClass, axis=0)

w_min_dist = stackedClasses_means - shamefulClass_mean
w_norm_min_dist = w_min_dist/np.linalg.norm(w_min_dist)

projClass1 = np.matmul(stackedClasses, w_norm_min_dist)
projClass2 = np.matmul(shamefulClass, w_norm_min_dist)

# Определяем диапазон значений для оси X
x_vals = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 1000)

# средние и дисперсии проекций
projClass1_M = np.mean(projClass1)
projClass2_M = np.mean(projClass2)
projClass1_D = np.sqrt(np.var(projClass1))
projClass2_D = np.sqrt(np.var(projClass2))

#генерируем нормальное распределение по средним и дисперсиям
ogib1 = stats.norm.pdf(x_vals,projClass1_M, projClass1_D)
ogib2 = stats.norm.pdf(x_vals, projClass2_M, projClass2_D)

#находим точки пересечений, где разница меняет знак
difference = ogib1 - ogib2
sign_change_idx = np.where(np.diff(np.sign(difference)))[0]
threshold = x_vals[sign_change_idx][0]

# Отображаем гистограммы, огибающие и точки пересечения
plt.figure()
plt.hist(projClass1, bins=20, density=True, color='blue', edgecolor='black', alpha=0.5, label='1+2')
plt.hist(projClass2, bins=20, density=True, color='orange', edgecolor='black', alpha=0.5, label='3')
plt.plot(x_vals, ogib1, color='red', linewidth=2, label="Огибающая 1+2")
plt.plot(x_vals, ogib2, color='blue', linewidth=2, label="Огибающая 3")
plt.xlim(left=min(projClass1.min(), projClass2.min()), right=max(projClass1.max(), projClass2.max()))
plt.axvline(x_vals[sign_change_idx], label=f'Порог: x= {threshold:.4f}')  # красные точки для пересечения
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

x_vals_2 = np.linspace(min(projClass1_2.min(), projClass2_2.min()), max(projClass1_2.max(), projClass2_2.max()), 1000)

projClass1_M_2 = np.mean(projClass1_2)
projClass2_M_2 = np.mean(projClass2_2)
projClass1_D_2 = np.sqrt(np.var(projClass1_2))
projClass2_D_2 = np.sqrt(np.var(projClass2_2))

ogib1_2 = stats.norm.pdf(x_vals_2,projClass1_M_2, projClass1_D_2)
ogib2_2 = stats.norm.pdf(x_vals_2, projClass2_M_2, projClass2_D_2)

difference_2 = ogib1_2 - ogib2_2
sign_change_idx_2 = np.where(np.diff(np.sign(difference_2)))[0]
threshold_2 = x_vals_2[sign_change_idx_2][0]

# Отображаем гистограммы, огибающие и точки пересечения
plt.figure()
plt.hist(projClass1_2, bins=20, density=True, color='blue', edgecolor='black', alpha=0.5, label='1')
plt.hist(projClass2_2, bins=20, density=True, color='orange', edgecolor='black', alpha=0.5, label='2') 
plt.plot(x_vals_2, ogib1_2, color='red', linewidth=2, label="Огибающая 1")
plt.plot(x_vals_2, ogib2_2, color='blue', linewidth=2, label="Огибающая 2")
plt.xlim(left=min(projClass1_2.min(), projClass2_2.min()), right=max(projClass1_2.max(), projClass2_2.max()))
plt.axvline(x_vals_2[sign_change_idx_2], label=f'Порог: x= {threshold_2:.4f}')  # красные точки для пересечения
plt.xlabel('Значение')
plt.ylabel('Плотность')
plt.title('Гистограммы с пересечением огибающих')
plt.legend()
plt.show()

# построение ROC-кривой
all_data_2 = np.delete(all_data, 0, axis=1)
r1 = np.matmul(all_data_2, w_norm_min_dist) - threshold

ones = np.ones(30) * 1    # Массив из 30 единиц
twos = np.ones(30) * 2     # Массив из 30 двоек
threes = np.ones(30) * 3   # Массив из 30 троек
metki1_train = np.concatenate((ones, twos, threes))

def calculate_metrics(data, w, threshold, labels):
    # Вычисляем проекции
    projections = np.matmul(data, w) - threshold

    # Классификация
    predicted_labels = (projections > 0).astype(int)

    # Преобразуем метки 1 и 2 в класс "1", а остальные в "0"
    true_positive_labels = (labels == 1) | (labels == 2)
    true_negative_labels = ~true_positive_labels

    tp = np.sum(predicted_labels[true_positive_labels] == 1)
    fn = np.sum(predicted_labels[true_positive_labels] == 0)
    tn = np.sum(predicted_labels[true_negative_labels] == 0)
    fp = np.sum(predicted_labels[true_negative_labels] == 1)

    # Защита от деления на 0
    se = tp / (tp + fn) if (tp + fn) > 0 else 0  # Чувствительность
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Специфичность
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0  # Точность

    return se, sp

# Генерация порогов для projClass1 и projClass2
thresholds_proj1 = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 1000)
se_proj1, sp_proj1 = [], []

# Расчёт метрик для projClass1
for threshold in thresholds_proj1:
    se, sp = calculate_metrics(all_data_2, w_norm_min_dist, threshold, metki1_train)
    se_proj1.append(se)
    sp_proj1.append(1 - sp)

# Генерация порогов для projClass2
thresholds_proj2 = np.linspace(min(projClass1_2.min(), projClass2_2.min()), max(projClass1_2.max(), projClass2_2.max()), 1000)
se_proj2, sp_proj2 = [], []

# Расчёт метрик для projClass2
for threshold in thresholds_proj2:
    se, sp = calculate_metrics(all_data_2[:60], w_norm_min_dist_2, threshold, metki1_train[:60])
    se_proj2.append(se)
    sp_proj2.append(1 - sp)

# Построение ROC-кривой для projClass1
plt.figure(figsize=(10, 6))
plt.plot(sp_proj1, se_proj1, label="ROC Curve for projClass1", color="blue")
plt.xlabel("1 - Specificity (FPR)")
plt.ylabel("Sensitivity (TPR)")
plt.title("ROC-кривая для projClass1")
plt.legend()
plt.grid()
#plt.show()

# Параметры гауссовых распределений для двух классов
mu1, sigma1 = projClass1_M, projClass1_D  # Класс 1
mu0, sigma0 = projClass2_M, projClass2_D  # Класс 0

# Пороги для классификации
thresholds = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 1000)
tpr = []
fpr = []

for T in thresholds:
    # Рассчитаем TPR и FPR для текущего порога
    tpr_value = 1 - stats.norm.cdf(T, mu1, sigma1)  # Интеграл для класса 1
    fpr_value = 1 - stats.norm.cdf(T, mu0, sigma0)  # Интеграл для класса 0
    
    tpr.append(tpr_value)
    fpr.append(fpr_value)

# Построение ROC-кривой
plt.plot(fpr, tpr, label="ROC Curve", color='green')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая методом Гаусса")
plt.legend()
plt.show()

# Функция для расчёта SE, SP и ACC
def calculate_metrics2(data, w, threshold, labels):
    # Вычисляем проекции
    projections = np.matmul(data, w) - threshold

    # Классификация
    predicted_labels = (projections > 0).astype(int)

    # Преобразуем метки: 1 класс остаётся 1, 2 класс - 0
    true_positive_labels = (labels == 1)
    true_negative_labels = (labels == 2)


    tp = np.sum(predicted_labels[true_positive_labels] == 1)
    fn = np.sum(predicted_labels[true_positive_labels] == 0)
    tn = np.sum(predicted_labels[true_negative_labels] == 0)
    fp = np.sum(predicted_labels[true_negative_labels] == 1)

    # Защита от деления на 0
    se = tp / (tp + fn) if (tp + fn) > 0 else 0  # Чувствительность
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Специфичность

    return se, sp

# Генерация порогов
thresholds_proj2_2 = np.linspace(min(projClass1_2.min(), projClass2_2.min()), max(projClass1_2.max(), projClass2_2.max()), 10000)
se_proj2_2, sp_proj2_2 = [], []

# Расчёт метрик для projClass1_2 и projClass2_2
for threshold in thresholds_proj2_2:
    se, sp = calculate_metrics2(np.vstack((class1, class2)), w_norm_min_dist_2, threshold, np.hstack((np.ones(len(class1)), np.ones(len(class2)) * 2)))
    se_proj2_2.append(se)
    sp_proj2_2.append(1 - sp)

# Построение ROC-кривой
plt.figure(figsize=(10, 6))
plt.plot(sp_proj2_2, se_proj2_2, label="ROC Curve for projClass1_2 and projClass2_2", color="blue")
plt.xlabel("1 - Specificity (FPR)")
plt.ylabel("Sensitivity (TPR)")
plt.title("ROC-кривая для projClass1_2 и projClass2_2")
plt.legend()
plt.grid()
#plt.show()

# Параметры гауссовых распределений для двух классов
mu1, sigma1 = projClass1_M_2, projClass1_D_2  # Класс 1
mu0, sigma0 = projClass2_M_2, projClass2_D_2  # Класс 0

# Пороги для классификации
thresholds = np.linspace(min(projClass1_2.min(), projClass2_2.min()), max(projClass1_2.max(), projClass2_2.max()), 10000)
tpr = []
fpr = []

for T in thresholds:
    # Рассчитаем TPR и FPR для текущего порога
    tpr_value = 1 - stats.norm.cdf(T, mu1, sigma1)  # Интеграл для класса 1
    fpr_value = 1 - stats.norm.cdf(T, mu0, sigma0)  # Интеграл для класса 0
    
    tpr.append(tpr_value)
    fpr.append(fpr_value)

# Построение ROC-кривой
plt.plot(fpr, tpr, label="ROC Curve", color='green')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая методом Гаусса")
plt.legend()
plt.show()

#####################AAAAAAAAAAAAAAAAAAAAAAAAAAAAA НЕ ТРОГАТЬ, УБЬЕТ - пытались проверить с помощью функции
# Вычисление ROC-кривой
'''
yTest = np.array([0]*30 + [1]*30)
#fpr, tpr, thresholds = roc_curve(yTest, predicted_labels)
# Вычисление площади под кривой (AUC)
#roc_auc = auc(fpr, tpr)

# Построение ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2) #label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Линия случайной модели
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA НЕ ТРОГАТЬ, УБЬЕТ
'''
#########################################################################################################

'''
ЛДФ - Фишер, сначала посмотрели МГК, выделили "позорный класс"
'''

sum_covariance = np.cov(stackedClasses, rowvar=0) + np.cov(shamefulClass, rowvar=0)

w_ldf = np.matmul(np.linalg.inv(sum_covariance), (stackedClasses_means - shamefulClass_mean))
w_norm_ldf = w_ldf/np.linalg.norm(w_ldf)

projClass1 = np.matmul(stackedClasses, w_norm_ldf)
projClass2 = np.matmul(shamefulClass, w_norm_ldf)

# Определяем диапазон значений для оси X
x_vals = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 1000)

# средние и дисперсии проекций
projClass1_M = np.mean(projClass1)
projClass2_M = np.mean(projClass2)
projClass1_D = np.sqrt(np.var(projClass1))
projClass2_D = np.sqrt(np.var(projClass2))

#генерируем нормальное распределение по средним и дисперсиям
ogib1 = stats.norm.pdf(x_vals,projClass1_M, projClass1_D)
ogib2 = stats.norm.pdf(x_vals, projClass2_M, projClass2_D)

#находим точки пересечений, где разница меняет знак
difference = ogib1 - ogib2
sign_change_idx = np.where(np.diff(np.sign(difference)))[0]
threshold = x_vals[sign_change_idx][0]

# Отображаем гистограммы, огибающие и точки пересечения
plt.figure()
plt.hist(projClass1, bins=20, density=True, color='blue', edgecolor='black', alpha=0.5, label='1+2')
plt.hist(projClass2, bins=20, density=True, color='orange', edgecolor='black', alpha=0.5, label='3')
plt.plot(x_vals, ogib1, color='red', linewidth=2, label="Огибающая 1+2")
plt.plot(x_vals, ogib2, color='blue', linewidth=2, label="Огибающая 3")
plt.xlim(left=min(projClass1.min(), projClass2.min()), right=max(projClass1.max(), projClass2.max()))
plt.axvline(x_vals[sign_change_idx], label=f'Порог: x= {threshold:.4f}')  # красные точки для пересечения
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

sum_covariance_2 = np.cov(class1, rowvar=0) + np.cov(class2, rowvar=0)
#
w_ldf_2 = np.matmul(np.linalg.inv(sum_covariance_2), (class1_mean - class2_mean)) #
w_norm_ldf_2 = w_ldf_2/np.linalg.norm(w_ldf_2)

projClass1_2 = np.matmul(class1, w_norm_ldf_2)
projClass2_2 = np.matmul(class2, w_norm_ldf_2)

x_vals_2 = np.linspace(min(projClass1_2.min(), projClass2_2.min()), max(projClass1_2.max(), projClass2_2.max()), 1000)

projClass1_M_2 = np.mean(projClass1_2)
projClass2_M_2 = np.mean(projClass2_2)
projClass1_D_2 = np.sqrt(np.var(projClass1_2))
projClass2_D_2 = np.sqrt(np.var(projClass2_2))

ogib1_2 = stats.norm.pdf(x_vals_2,projClass1_M_2, projClass1_D_2)
ogib2_2 = stats.norm.pdf(x_vals_2, projClass2_M_2, projClass2_D_2)

difference_2 = ogib1_2 - ogib2_2
sign_change_idx_2 = np.where(np.diff(np.sign(difference_2)))[0]
threshold_2 = x_vals_2[sign_change_idx_2][0]

# Отображаем гистограммы, огибающие и точки пересечения
plt.figure()
plt.hist(projClass1_2, bins=20, density=True, color='blue', edgecolor='black', alpha=0.5, label='1')
plt.hist(projClass2_2, bins=20, density=True, color='orange', edgecolor='black', alpha=0.5, label='2') 
plt.plot(x_vals_2, ogib1_2, color='red', linewidth=2, label="Огибающая 1")
plt.plot(x_vals_2, ogib2_2, color='blue', linewidth=2, label="Огибающая 2")
plt.xlim(left=min(projClass1_2.min(), projClass2_2.min()), right=max(projClass1_2.max(), projClass2_2.max()))
plt.axvline(x_vals_2[sign_change_idx_2], label=f'Порог: x= {threshold_2:.4f}')  # красные точки для пересечения
plt.xlabel('Значение')
plt.ylabel('Плотность')
plt.title('Гистограммы с пересечением огибающих')
plt.legend()
plt.show()

# построение ROC-кривой
all_data_2 = np.delete(all_data, 0, axis=1)
r1 = np.matmul(all_data_2, w_norm_ldf) - threshold

ones = np.ones(30) * 1    # Массив из 30 единиц
twos = np.ones(30) * 2     # Массив из 30 двоек
threes = np.ones(30) * 3   # Массив из 30 троек
metki1_train = np.concatenate((ones, twos, threes))

def calculate_metrics(data, w, threshold, labels):
    # Вычисляем проекции
    projections = np.matmul(data, w) - threshold

    # Классификация
    predicted_labels = (projections > 0).astype(int)

    # Преобразуем метки 1 и 2 в класс "1", а остальные в "0"
    true_positive_labels = (labels == 1) | (labels == 2)
    true_negative_labels = ~true_positive_labels

    tp = np.sum(predicted_labels[true_positive_labels] == 1)
    fn = np.sum(predicted_labels[true_positive_labels] == 0)
    tn = np.sum(predicted_labels[true_negative_labels] == 0)
    fp = np.sum(predicted_labels[true_negative_labels] == 1)

    # Защита от деления на 0
    se = tp / (tp + fn) if (tp + fn) > 0 else 0  # Чувствительность
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Специфичность
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0  # Точность

    return se, sp, acc

# Генерация порогов для projClass1 и projClass2
thresholds_proj1 = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 1000)
se_proj1, sp_proj1 = [], []

# Расчёт метрик для projClass1
for threshold in thresholds_proj1:
    se, sp, _ = calculate_metrics(all_data_2, w_norm_ldf, threshold, metki1_train)
    se_proj1.append(se)
    sp_proj1.append(1 - sp)

# Генерация порогов для projClass2
thresholds_proj2 = np.linspace(min(projClass1_2.min(), projClass2_2.min()), max(projClass1_2.max(), projClass2_2.max()), 1000)
se_proj2, sp_proj2 = [], []

# Расчёт метрик для projClass2
for threshold in thresholds_proj2:
    se, sp, _ = calculate_metrics(all_data_2[:60], w_norm_ldf_2, threshold, metki1_train[:60])
    se_proj2.append(se)
    sp_proj2.append(1 - sp)

# Построение ROC-кривой для projClass1
plt.figure(figsize=(10, 6))
plt.plot(sp_proj1, se_proj1, label="ROC Curve for projClass1", color="blue")
plt.xlabel("1 - Specificity (FPR)")
plt.ylabel("Sensitivity (TPR)")
plt.title("ROC-кривая для projClass1")
plt.legend()
plt.grid()
#plt.show()

# Параметры гауссовых распределений для двух классов
mu1, sigma1 = projClass1_M, projClass1_D  # Класс 1
mu0, sigma0 = projClass2_M, projClass2_D  # Класс 0

# Пороги для классификации
thresholds = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 1000)
tpr = []
fpr = []

for T in thresholds:
    # Рассчитаем TPR и FPR для текущего порога
    tpr_value = 1 - stats.norm.cdf(T, mu1, sigma1)  # Интеграл для класса 1
    fpr_value = 1 - stats.norm.cdf(T, mu0, sigma0)  # Интеграл для класса 0
    
    tpr.append(tpr_value)
    fpr.append(fpr_value)

# Построение ROC-кривой
plt.plot(fpr, tpr, label="ROC Curve", color='green')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая методом Гаусса")
plt.legend()
plt.show()

# Функция для расчёта SE, SP и ACC
def calculate_metrics2(data, w, threshold, labels):
    # Вычисляем проекции
    projections = np.matmul(data, w) - threshold

    # Классификация
    predicted_labels = (projections > 0).astype(int)

    # Преобразуем метки: 1 класс остаётся 1, 2 класс - 0
    true_positive_labels = (labels == 1)
    true_negative_labels = (labels == 2)

    tp = np.sum(predicted_labels[true_positive_labels] == 1)
    fn = np.sum(predicted_labels[true_positive_labels] == 0)
    tn = np.sum(predicted_labels[true_negative_labels] == 0)
    fp = np.sum(predicted_labels[true_negative_labels] == 1)

    # Защита от деления на 0
    se = tp / (tp + fn) if (tp + fn) > 0 else 0  # Чувствительность
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Специфичность

    return se, sp

# Генерация порогов
thresholds_proj2_2 = np.linspace(min(projClass1_2.min(), projClass2_2.min()), max(projClass1_2.max(), projClass2_2.max()), 10000)
se_proj2_2, sp_proj2_2 = [], []

# Расчёт метрик для projClass1_2 и projClass2_2
for threshold in thresholds_proj2_2:
    se, sp = calculate_metrics2(np.vstack((class1, class2)), w_norm_ldf_2, threshold, np.hstack((np.ones(len(class1)), np.ones(len(class2)) * 2)))
    se_proj2_2.append(se)
    sp_proj2_2.append(1 - sp)

# Построение ROC-кривой
plt.figure(figsize=(10, 6))
plt.plot(sp_proj2_2, se_proj2_2, label="ROC Curve for projClass1_2 and projClass2_2", color="blue")
plt.xlabel("1 - Specificity (FPR)")
plt.ylabel("Sensitivity (TPR)")
plt.title("ROC-кривая для projClass1_2 и projClass2_2")
plt.legend()
plt.grid()
#plt.show()

# Параметры гауссовых распределений для двух классов
mu1, sigma1 = projClass1_M_2, projClass1_D_2  # Класс 1
mu0, sigma0 = projClass2_M_2, projClass2_D_2  # Класс 0

# Пороги для классификации
thresholds = np.linspace(min(projClass1_2.min(), projClass2_2.min()), max(projClass1_2.max(), projClass2_2.max()), 10000)
tpr = []
fpr = []

for T in thresholds:
    # Рассчитаем TPR и FPR для текущего порога
    tpr_value = 1 - stats.norm.cdf(T, mu1, sigma1)  # Интеграл для класса 1
    fpr_value = 1 - stats.norm.cdf(T, mu0, sigma0)  # Интеграл для класса 0
    
    tpr.append(tpr_value)
    fpr.append(fpr_value)

# Построение ROC-кривой
plt.plot(fpr, tpr, label="ROC Curve", color='green')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая методом Гаусса")
plt.legend()
plt.show()

'''
Критерий Фишера (множественный анализ)
'''

def gaussian(class1, class2, w):
    projClass1 = np.matmul(class1, w)
    projClass2 = np.matmul(class2, w)
    x_vals = np.linspace(min(projClass1.min(), projClass2.min()), max(projClass1.max(), projClass2.max()), 1000)
    ogib1 = stats.norm.pdf(x_vals,projClass1_M, projClass1_D)
    ogib2 = stats.norm.pdf(x_vals, projClass2_M, projClass2_D)
    difference = ogib1 - ogib2
    sign_change_idx = np.where(np.diff(np.sign(difference)))[0]
    threshold = x_vals[sign_change_idx][0]
    return threshold, sign_change_idx

#вызов функции get_perpendicular_line_in_point(threshold, sign_change_idx, x, -norm_vector[0]/norm_vector[1], -intercept/norm_vector[1])
def get_perpendicular_line_in_point(pointX, y_idx, x, k, b):
    y = k*x + b
    a = 1/k
    y_thr = y[y_idx]
    k_p = -(1/a)
    b_p = (1/a)*pointX + y_thr
    y_1 = -(1/a)*(x_vals - pointX) + y_thr #уравнение перпендикулярной прямой
    return y_1, k_p, b_p

# Создаем метки классов
labels = np.array([1]*30 + [2]*30 + [3]*30)  # 1 - НР, 2 - ЖТ, 3 - ФЖ

# Разделяем данные на классы
class_1 = all_data_2[:30]  # НР
class_2 = all_data_2[30:60]  # ЖТ
class_3 = all_data_2[60:]  # ФЖ

# Средние значения для каждого класса
mean_1 = np.mean(class_1, axis=0)
mean_2 = np.mean(class_2, axis=0)
mean_3 = np.mean(class_3, axis=0)

# Общая средняя
overall_mean = np.mean(all_data_2, axis=0)

# Внутриклассовая ковариационная матрица
Sw = np.zeros((all_data_2.shape[1], all_data_2.shape[1]))
for c, mean in zip([class_1, class_2, class_3], [mean_1, mean_2, mean_3]):
    Sw += np.cov(c, rowvar=False) * (c.shape[0] - 1)  # Ковариация внутри классов

# Межклассовая ковариационная матрица
Sb = np.zeros((all_data_2.shape[1], all_data_2.shape[1]))
for mean in [mean_1, mean_2, mean_3]:
    n_c = 30  # Количество объектов в каждом классе
    mean_diff = (mean - overall_mean).reshape(-1, 1)
    Sb += n_c * (mean_diff @ mean_diff.T)

# Добавляем регуляризацию к Sw
regularization_param = 1e-6  # Маленькая регуляризация для стабильности
Sw += regularization_param * np.eye(Sw.shape[0])  # Добавляем λI

# Собственные векторы и значения
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) @ Sb)

# Сортируем собственные векторы по убыванию собственных значений
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Выбираем два первых дискриминантных вектора
w1 = eigenvectors[:, 0]
w2 = eigenvectors[:, 1]

# Проекция данных на дискриминантные оси
proj_data = all_data_2 @ np.column_stack((w1, w2))

# Вычисляем центры каждого класса в пространстве дискриминантных осей
center_1 = np.mean(proj_data[:30], axis=0)  # Центр НР
center_2 = np.mean(proj_data[30:60], axis=0)  # Центр ЖТ
center_3 = np.mean(proj_data[60:], axis=0)  # Центр ФЖ

# Функция для нахождения коэффициентов разделяющей прямой
def separating_line(center_a, center_b):
    # Вектор между центрами классов
    delta = center_b - center_a
    
    # Средняя точка между центрами
    midpoint = (center_a + center_b) / 2  
    
    # Нормальный вектор прямой (перпендикулярен линии между центрами)
    normal_vector = np.array([-delta[1], delta[0]])
    
    # Уравнение прямой: normal_vector[0] * x + normal_vector[1] * y + intercept = 0
    intercept = -np.dot(normal_vector, midpoint)
    
    return normal_vector, intercept

# Прямые для каждой пары классов
normal_12, intercept_12 = separating_line(center_1, center_2)
normal_13, intercept_13 = separating_line(center_1, center_3)
normal_23, intercept_23 = separating_line(center_2, center_3)

# Функция для отображения прямой
def plot_line(normal_vector, intercept, label, color):
    x = np.linspace(-4, 4, 100)  # Диапазон для x
    y = -(normal_vector[0] * x + intercept) / normal_vector[1]  # Уравнение прямой
    plt.plot(x, y, label=label, color=color, linestyle='--')

# Визуализация
plt.figure(figsize=(8, 6))

# Отображение данных
plt.scatter(proj_data[:30, 0], proj_data[:30, 1], label='НР', alpha=0.7)
plt.scatter(proj_data[30:60, 0], proj_data[30:60, 1], label='ЖТ', alpha=0.7)
plt.scatter(proj_data[60:, 0], proj_data[60:, 1], label='ФЖ', alpha=0.7)

plt.xlabel('1-я дискриминантная ось')
plt.ylabel('2-я дискриминантная ось')
plt.title('Распределение объектов и разделяющие прямые')
plt.legend()
plt.grid()
plt.show()