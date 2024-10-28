import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import gaussian_kde
# Путь к файлу Excel
file_path = 'C:/Users/infor/OneDrive/Рабочий стол/Универ/11 сем/ТиСПР/Практика/Варианты_9503/Вар_1/10 Гц, шаг 0.7, 1 выборка.xlsx'

# Чтение данных из файла Excel
# словарь датафреймов
data = pd.read_excel(file_path, None)
data_norm = copy.deepcopy(data)

array = np.zeros([30, 12, 3])
i = 0

# Деление каждой строки на ее нулевой элемент
for x in ["НР", "ЖТ", "ФЖ"]:
    data_norm[x] = data[x].apply(lambda row: row / row.iloc[0], axis=1)
    array[:, :, i] = data_norm[x].to_numpy()
    i = i + 1

# сделали столбец меток для каждого ритма (1 - НР, 2 - ЖТ, 3 - ФЖ)
array[:, 0, 1] = 2.0
array[:, 0, 2] = 3.0
#print(array[:, 0, 2])

X_train = np.empty((0, 12))
X_test = np.empty((0, 12))
Y_train = np.empty((0, 1))
Y_test = np.empty((0, 1))

for o in range(3):
    X_train = np.vstack((X_train, array[:15, :, o]))
    X_test = np.vstack((X_test, array[15:, :, o]))

Y_train = X_train[:, 0]
Y_test = X_test[:, 0]

X_train = np.delete(X_train, 0, axis=1)
X_test = np.delete(X_test, 0, axis=1)

#к ближайших соседей
model = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

#оценка близости для k ближайших соседей
model2 = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean')
model2.fit(X_train, Y_train)
predictions2 = model2.predict(X_test)

print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))

print(classification_report(Y_test, predictions2))
print(confusion_matrix(Y_test, predictions2))

#ЛДФ

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